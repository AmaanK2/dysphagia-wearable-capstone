import io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import stft
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn

labels=['cough','speak','swallow','throatclear']
out=Path('output/throat_baseline')
def load(split):
    data=[]; targets=[]
    with zipfile.ZipFile(f'data/throat/{split}.zip') as z:
        for name in z.namelist():
            label=Path(name).stem
            if label in labels:
                x=np.load(io.BytesIO(z.read(name)),allow_pickle=False).astype('float32')
                data.append(x)
                targets.extend([labels.index(label)]*len(x))
    return np.concatenate(data),np.array(targets)
def transform(x):
    _,_,s=stft(x,fs=6400,nperseg=256,noverlap=192)
    return np.log10(abs(s)**2+1e-12).astype('float32')[:,None]
if __name__=='__main__':
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(3)
    x,y=load('train'); train,val=train_test_split(np.arange(len(y)),test_size=.15,stratify=y,random_state=2026)
    x=transform(x); mean=x[train].mean(); std=x[train].std(); x=torch.from_numpy((x-mean)/std)
    y=torch.from_numpy(y)
    model=nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                        nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                        nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Dropout(.2),nn.Linear(768,4))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    best=-1
    for epoch in range(1,13):
        model.train()
        for idx in np.array_split(train[torch.randperm(len(train)).numpy()],len(train)//64):
            opt.zero_grad(); loss=nn.functional.cross_entropy(model(x[idx]),y[idx]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            pred=torch.cat([model(batch).argmax(1) for batch in x[val].split(128)])
        score=float((pred==y[val]).float().mean())
        print('Epoch',epoch,'validation accuracy',round(score,4),flush=True)
        if score>best:
            best=score; checkpoint={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(checkpoint)
    raw,y=load('test'); x=torch.from_numpy((transform(raw)-mean)/std)
    with torch.no_grad():
        pred=torch.cat([model(batch).argmax(1) for batch in x.split(128)]).numpy()
    result={'labels':labels,'best_clip_validation_accuracy':best,
            'report':classification_report(y,pred,labels=list(range(4)),target_names=labels,output_dict=True),
            'confusion_matrix':confusion_matrix(y,pred,labels=list(range(4))).tolist()}
    (out/'cnn_metrics.json').write_text(json.dumps(result,indent=2))
    torch.save({'state_dict':checkpoint,'mean':float(mean),'std':float(std),'labels':labels},out/'cnn.pt')
    print(result)
