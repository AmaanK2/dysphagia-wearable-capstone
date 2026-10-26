"""Predict annotated-swallow intervals with the saved PyTorch checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from cnn import SwallowCNN, preprocess, INPUT_FS, OUTPUT_FS
from train import events

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--recording',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('prediction_cnn'))
args=p.parse_args();torch.set_num_threads(2)
checkpoint=torch.load(args.model,map_location='cpu',weights_only=True)
x=np.load(args.recording,allow_pickle=False)
if x.ndim!=2 or x.shape[1]!=4 or len(x)<2000 or not np.isfinite(x).all():
 raise ValueError('Expected finite N x 4 recording, >=0.5 seconds, sampled at 4 kHz, microphone/AP/SI/ML order')
v=preprocess(x)
v=np.clip((v-np.array(checkpoint['mean'],dtype=np.float32))/np.array(checkpoint['std'],dtype=np.float32),-20,20)
model=SwallowCNN().eval();model.load_state_dict(checkpoint['state_dict'])
with torch.no_grad():
 score,n=model(torch.from_numpy(v.T.copy())[None],torch.tensor([len(v)]))
 score=score[0,:int(n[0])].sigmoid().numpy()
centers=np.arange(500,len(v)-500+1,200)/INPUT_FS
score=np.interp(centers,np.arange(len(score))/OUTPUT_FS,score)
intervals=events(centers,score,checkpoint['threshold'],len(x)/4000)
args.output.mkdir(parents=True,exist_ok=True)
with (args.output/'scores.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['center_seconds','uncalibrated_model_score']);w.writerows(zip(centers,score))
(args.output/'intervals.json').write_text(json.dumps({'intervals_seconds':intervals,'threshold':checkpoint['threshold'],
 'scope':'Offline research prototype; no diagnosis or aspiration-risk prediction'},indent=2))
print('Predicted intervals:',len(intervals))
