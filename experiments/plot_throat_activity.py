import json
from pathlib import Path
import numpy as np
from scipy.signal import stft,welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_throat_cnn import load
x,y=load('train')
labels=['Cough','Speech','Swallow','Throat clear']
colors=['#bb5a31','#4d65a8','#138475','#795a9c']
fig,axs=plt.subplots(4,3,figsize=(13,11),layout='constrained')
fig.suptitle('What four throat activities look like',fontsize=23,weight='bold')
for k,(name,col) in enumerate(zip(labels,colors)):
    activity=x[y==k]
    f,p=welch(activity,fs=6400,nperseg=256,axis=-1)
    p=p/(p.sum(axis=1,keepdims=True)+1e-20)
    rms=np.sqrt(np.mean(activity**2,axis=1))
    i=np.argmin(abs(rms-np.median(rms)))
    clip=activity[i]; sec=np.arange(len(clip))/6400
    axs[k,0].plot(sec,clip/(abs(clip).max()+1e-20),color=col,lw=.9)
    axs[k,0].set_ylabel(name,fontsize=12,weight='bold',color=col)
    axs[k,0].set_ylim(-1.05,1.05)
    fs,t,z=stft(clip,fs=6400,nperseg=256,noverlap=192)
    db=20*np.log10(abs(z)/(abs(z).max()+1e-20)+1e-8)
    axs[k,1].pcolormesh(t,fs,db,vmin=-50,vmax=0,cmap='magma',shading='auto')
    axs[k,1].set_ylim(0,2000)
    for j,c in enumerate(colors):
        if k==0:
            freq,power=welch(x[y==j],fs=6400,nperseg=256,axis=-1)
            frac=power/(power.sum(axis=1,keepdims=True)+1e-20)
            med=np.median(frac,axis=0)
            axs[k,2].plot(freq[1:],10*np.log10(med[1:]+1e-9),color=c,label=labels[j])
    if k==0:
        axs[k,2].legend(fontsize=8,frameon=False)
        axs[k,2].set_xlabel('Frequency (Hz)')
        axs[k,2].set_ylabel('Median relative power (dB)')
        axs[k,2].set_xlim(0,2000)
    else:
        axs[k,2].axis('off')
for ax,title in zip(axs[0],['Example waveform','Example spectrogram','Where energy overlaps']): ax.set_title(title,weight='bold')
axs[-1,0].set_xlabel('Seconds · amplitude scaled per clip')
axs[-1,1].set_xlabel('Seconds · brightness scaled per clip')
for ax in axs[:,1]: ax.set_ylabel('Frequency (Hz)')
rf=json.loads(Path('output/throat_baseline/metrics.json').read_text())
cnn_file=Path('output/throat_baseline/cnn_metrics.json')
if cnn_file.exists():
    cnn=json.loads(cnn_file.read_text())
    ax=axs[1,2];ax.axis('on')
    values=np.arange(4)
    ax.barh(values+.15,[rf['report'][key]['f1-score']*100 for key in rf['labels']],height=.29,label='Random forest',color='#8798a6')
    ax.barh(values-.15,[cnn['report'][key]['f1-score']*100 for key in rf['labels']],height=.29,label='Spectrogram CNN',color='#138475')
    ax.set_yticks(values,labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Test F1 (%)');ax.legend(fontsize=8,loc='lower right',frameon=False)
ax=axs[2,2];ax.text(0,.86,'Read the colors',fontsize=13,weight='bold',transform=ax.transAxes)
ax.text(0,.65,'Brighter regions show stronger vibrations\nat that time and frequency. Individual clips\ncan look different within the same class.',fontsize=11,transform=ax.transAxes,va='top')
ax=axs[3,2];ax.text(0,.86,'What the dataset covers',fontsize=13,weight='bold',transform=ax.transAxes)
ax.text(0,.65,'Song et al. throat-vibration training clips.\nEach example has median RMS for its class.\nNo labeled head-turn class in the release.',fontsize=11,transform=ax.transAxes,va='top')
figure_path = Path('output/throat_baseline/throat_activity_signals.png')
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=140)
