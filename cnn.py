"""Small offline PyTorch temporal segmenter on the frozen participant split."""
import argparse
import copy
import hashlib
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfiltfilt, resample_poly
from train import event_metrics, window_metrics, bootstrap_interval, ANNOTATION_NOTICE

INPUT_FS = 2000
OUTPUT_FS = 50
STRIDE = 40
SEED = 42


def zero_padding(x, lengths):
    mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return x * mask[:, None, :]


class SwallowCNN(nn.Module):
    """~40k parameters. Symmetric context; not a causal/real-time model."""
    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList([nn.Conv1d(4, 16, 31, stride=5, padding=15),
                                   nn.Conv1d(16, 24, 9, stride=4, padding=4),
                                   nn.Conv1d(24, 32, 7, stride=2, padding=3)])
        self.temporal = nn.ModuleList([nn.Conv1d(32, 32, 5, padding=2*d, dilation=d)
                                       for d in [1, 2, 4, 8, 16]])
        self.dropout = nn.Dropout(.1)
        self.head = nn.Conv1d(32, 1, 1)

    def forward(self, x, lengths):
        for conv in self.stem:
            lengths = (lengths + conv.stride[0] - 1) // conv.stride[0]
            x = zero_padding(torch.nn.functional.silu(conv(x)), lengths)
        for conv in self.temporal:
            x = zero_padding(x + self.dropout(torch.nn.functional.silu(conv(x))), lengths)
        return self.head(x).squeeze(1), lengths


def preprocess(x):
    result = np.empty_like(x)
    for c in range(4):
        low, high = (20, 800) if c == 0 else (5, 500)
        result[:, c] = sosfiltfilt(butter(4, [low, high], fs=4000, btype='bandpass', output='sos'), x[:, c])
    return resample_poly(result, 1, 2, axis=0).astype(np.float32)


def prepare_cache(signals, meta, cache):
    # Verify input archive identity against the manifest used by the baseline.
    h = hashlib.md5()
    with signals.open('rb') as f:
        while b := f.read(4*1024*1024):
            h.update(b)
    if h.hexdigest() != 'ea0e725848c6092f946b06198f293f0d':
        raise ValueError('This experiment requires the original Zenodo 4539695 signal archive')
    config = {'source_md5': h.hexdigest(), 'fs': INPUT_FS, 'filter': 'butter4_mic20-800_acc5-500_zero_phase_resample_poly_1_2',
              'records': {r['id']: r['sha256'] for r in meta}}
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Cache configuration mismatch: choose a fresh --cache directory')
    with zipfile.ZipFile(signals) as z:
        for i, r in enumerate(meta):
            path = cache/(r['id']+'.npy')
            if not path.exists():
                raw = z.read('signals/'+r['id']+'.npy')
                if hashlib.sha256(raw).hexdigest() != r['sha256']:
                    raise ValueError('Recording differs from baseline manifest')
                np.save(path, preprocess(np.load(io.BytesIO(raw), allow_pickle=False)))
            if (i+1) % 250 == 0:
                print('Prepared', i+1, '/', len(meta), flush=True)
    manifest.write_text(json.dumps(config))


def fit_normalization(records, cache):
    total = np.zeros(4); squares = np.zeros(4); count = 0
    for r in records:
        x = np.load(cache/(r['id']+'.npy'), allow_pickle=False).astype(np.float64)
        total += x.sum(axis=0); squares += (x*x).sum(axis=0); count += len(x)
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class Records(Dataset):
    def __init__(self, records, cache, mean, std):
        self.records, self.cache, self.mean, self.std = records, cache, mean, std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        x = np.load(self.cache/(r['id']+'.npy'), allow_pickle=False)
        x = np.clip((x-self.mean)/self.std, -20, 20).T.copy()
        t = np.arange((x.shape[1]+STRIDE-1)//STRIDE)/OUTPUT_FS
        y = np.zeros(len(t), dtype=np.float32)
        for a, b in r['intervals']:
            y[(t >= a) & (t < b)] = 1
        return torch.from_numpy(x), torch.from_numpy(y), r['id']


def collate(items):
    lengths = torch.tensor([x.shape[1] for x,y,k in items], dtype=torch.long)
    x = torch.zeros(len(items), 4, int(lengths.max()))
    y = torch.zeros(len(items), (int(lengths.max())+STRIDE-1)//STRIDE)
    for i,(a,b,k) in enumerate(items):
        x[i,:,:a.shape[1]] = a; y[i,:len(b)] = b
    return x, y, lengths, [k for a,b,k in items]


def loss_value(logits, target, lengths, positive_weight):
    mask = torch.arange(logits.shape[1], device=logits.device)[None,:] < lengths[:,None]
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=positive_weight)
    return (losses*mask).sum()/mask.sum()


@torch.no_grad()
def predict_records(model, loader, records, device):
    model.eval(); result = {}; by_id = {r['id']:r for r in records}
    for x,y,lengths,keys in loader:
        logits,out_lengths = model(x.to(device), lengths.to(device))
        probabilities = logits.sigmoid().cpu().numpy()
        for i,key in enumerate(keys):
            n = int(out_lengths[i]); r = by_id[key]
            # Evaluate both methods on the baseline's exact 0.1s center grid.
            result[key] = np.interp(r['times'],np.arange(n)/OUTPUT_FS,probabilities[i,:n])
    return result


def evaluate(records, predictions, threshold, iou=.3):
    """Efficient equivalent of baseline's evaluator, using its matching rules."""
    from train import events, match_events
    tp=fp=fn=0; seconds=0.; errors=[]; by_person={}
    for r in records:
        p=events(r['times'], predictions[r['id']], threshold, r['duration'])
        a,b,c,err=match_events(p,r['intervals'],iou)
        tp+=a;fp+=b;fn+=c;seconds+=r['duration'];errors+=err
        row=by_person.setdefault(r['participant'],[0,0,0,0.])
        row[0]+=a;row[1]+=b;row[2]+=c;row[3]+=r['duration']
    return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
            'tp':tp,'fp':fp,'fn':fn,'false_events_per_recording_minute':fp/(seconds/60),
            'matched_onset_mae_seconds':float(np.mean(errors)) if errors else None,
            'iou_threshold':iou,'participant_counts':by_person}


def best_threshold(records, predictions):
    return max((evaluate(records,predictions,float(t))['f1'],float(t)) for t in np.linspace(.05,.95,37))


def main():
    print(ANNOTATION_NOTICE, flush=True)
    p=argparse.ArgumentParser()
    p.add_argument('--signals',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,default=Path('experiments/cnn_v1'))
    p.add_argument('--cache',type=Path,default=Path('_cache/cnn_v1'))
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():
        raise ValueError('Choose a fresh output directory; existing experiments are preserved')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    meta=json.loads((args.baseline/'records.json').read_text())
    splits=json.loads((args.baseline/'participant_splits.json').read_text())
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(splits[a]) & set(splits[b])
    records={k:[r for r in meta if r['participant'] in v] for k,v in splits.items()}
    prepare_cache(args.signals,meta,args.cache)
    mean,std=fit_normalization(records['train'],args.cache)
    np.savez(args.output/'normalization.npz',mean=mean,std=std)
    datasets={k:Records(v,args.cache,mean,std) for k,v in records.items()}
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,collate_fn=collate,generator=generator)
    val_loader=DataLoader(datasets['validation'],batch_size=args.batch_size,collate_fn=collate)
    positive=negative=0
    for x,y,key in datasets['train']:
        positive+=float(y.sum());negative+=len(y)-float(y.sum())
    weight=torch.tensor(negative/positive,device=device)
    model=SwallowCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    config={'seed':SEED,'torch_version':str(torch.__version__),'input_fs':INPUT_FS,'output_fs':OUTPUT_FS,
            'model_parameters':sum(p.numel() for p in model.parameters()),'positive_weight':float(weight),
            'normalization_fit_participants':splits['train'],'max_epochs':args.epochs,'patience':args.patience,
            'batch_size':args.batch_size,'learning_rate':.001,'weight_decay':.0001,'threads':args.threads,
            'scope':'Offline annotated-swallow segmentation, not dysphagia/aspiration classification',
            'test_policy':'Fixed pilot test cohort already inspected for baseline; not a new blind external validation',
            'fairness_note':'Same participants, fixed filters, target grid and event matching; CNN uses longer temporal context than RF.',
            'split_sizes':{k:len(v) for k,v in splits.items()}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'participant_splits.json').write_text(json.dumps(splits,indent=2))
    history=[];best=-1.;best_epoch=0;checkpoint=None;threshold=None
    print('Parameters:',config['model_parameters'],'training positive weight:',float(weight),flush=True)
    for epoch in range(1,args.epochs+1):
        started=time.time();model.train();loss_sum=0.;batches=0
        for x,y,lengths,keys in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits,out_lengths=model(x.to(device),lengths.to(device))
            loss=loss_value(logits,y.to(device),out_lengths,weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach());batches+=1
        predictions=predict_records(model,val_loader,records['validation'],device)
        f1,t=best_threshold(records['validation'],predictions)
        row={'epoch':epoch,'training_loss_mean_batch':loss_sum/batches,'validation_event_f1':f1,
             'validation_threshold':t,'seconds':time.time()-started}
        history.append(row);print(json.dumps(row),flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        if f1>best+1e-8:
            best=f1;best_epoch=epoch;threshold=t
            checkpoint=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
            torch.save({'state_dict':checkpoint,'mean':mean.tolist(),'std':std.tolist(),'threshold':threshold,
                        'epoch':epoch,'config':config},args.output/'best.pt')
        if epoch-best_epoch>=args.patience:
            print('Early stopping on validation event F1',flush=True);break
    model.load_state_dict(checkpoint)
    test_loader=DataLoader(datasets['test'],batch_size=args.batch_size,collate_fn=collate)
    # Test data first used after checkpoint and threshold are frozen.
    test_predictions=predict_records(model,test_loader,records['test'],device)
    event=evaluate(records['test'],test_predictions,threshold)
    ids=np.concatenate([[r['id']]*len(r['times']) for r in records['test']])
    scores=np.concatenate([test_predictions[r['id']] for r in records['test']])
    y=[]
    for r in records['test']:
        t=np.array(r['times']);labels=np.zeros(len(t),dtype=np.uint8)
        for a,b in r['intervals']:labels[(t>=a)&(t<b)]=1
        y.extend(labels)
    result={'best_epoch':best_epoch,'threshold':threshold,'validation_event_f1':best,'test_event':event,
            'test_event_strict_iou_0_5':evaluate(records['test'],test_predictions,threshold,.5),
            'test_window':window_metrics(np.array(y),scores,threshold),
            'test_event_bootstrap_95_ci':bootstrap_interval(event['participant_counts'])}
    np.savez_compressed(args.output/'test_predictions.npz',ids=ids,scores=scores,y=np.array(y))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print('FINAL TEST:',json.dumps({k:v for k,v in event.items() if k!='participant_counts'}),flush=True)


if __name__=='__main__':
    main()
