"""Download the original public archives without checking them into Git."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

FILES={
 'labels.zip':('https://zenodo.org/records/4539695/files/labels.zip?download=1','5af27a544839b404f359e3d72b5026e2'),
 'signals.zip':('https://zenodo.org/records/4539695/files/signals.zip?download=1','ea0e725848c6092f946b06198f293f0d'),
}
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/raw/hrca'));args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
for name,(url,expected) in FILES.items():
 path=args.output/name
 if not path.exists():
  temp=path.with_suffix('.zip.part')
  print('Downloading',name,flush=True)
  urllib.request.urlretrieve(url,temp)
  temp.rename(path)
 h=hashlib.md5()
 with path.open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 if h.hexdigest()!=expected:raise RuntimeError(f'Checksum failed for {name}; remove it and download again')
 print(name,'verified',flush=True)
