"""Download the public datasets into data/ (git-ignored) and verify their checksums.

    python download_data.py            # Pittsburgh swallow dataset (Zenodo 4539695, ~1.4 GB) -> data/raw
    python download_data.py --throat   # throat-event dataset (Zenodo 12630148, ~158 MB) -> data/throat
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

DATASETS = {
    'swallow': (Path('data/raw'), {
        'labels.zip': ('https://zenodo.org/records/4539695/files/labels.zip?download=1',
                       '5af27a544839b404f359e3d72b5026e2'),
        'signals.zip': ('https://zenodo.org/records/4539695/files/signals.zip?download=1',
                        'ea0e725848c6092f946b06198f293f0d'),
    }),
    'throat': (Path('data/throat'), {
        'train.zip': ('https://zenodo.org/api/records/12630148/files/Training%20dataset.zip/content',
                      '73ea26bc7bde041ff0690094f7477421'),
        'test.zip': ('https://zenodo.org/api/records/12630148/files/Test%20dataset.zip/content',
                     'bc9aebdee8a00d21d1da968ab3bb8027'),
    }),
}


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as file:
        while block := file.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--throat', action='store_true', help='Download the throat-event dataset instead')
    parser.add_argument('--output', type=Path, help='Override the output folder')
    args = parser.parse_args()
    folder, files = DATASETS['throat' if args.throat else 'swallow']
    folder = args.output or folder
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in files.items():
        path = folder / name
        if not path.exists():
            print('Downloading', name, flush=True)
            partial = path.with_suffix('.part')
            urllib.request.urlretrieve(url, partial)
            partial.rename(path)
        if md5(path) != expected:
            raise RuntimeError(f'Checksum failed for {path}; delete it and download again')
        print(name, 'verified', flush=True)


if __name__ == '__main__':
    main()
