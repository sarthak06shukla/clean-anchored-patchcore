"""Download the bottle subset from a public MVTec AD mirror, with provenance."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = 'foersben/mvtec-ad'

def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--revision', default=None)
    parser.add_argument('--category', choices=['bottle','hazelnut','tile'], default='bottle')
    args = parser.parse_args()
    revision = args.revision or json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}'))['sha']
    entries = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{args.category}?recursive=true&limit=1000'))
    entries = [e for e in entries if e['type'] == 'file']
    def download(entry):
        path = args.root / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != entry['size']:
            payload = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}?download=true")
            if len(payload) != entry['size']:
                raise ValueError(f'Incorrect size: {path}')
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = entry.get('lfs', {}).get('oid')
        if expected and digest != expected:
            raise ValueError(f'Mirror LFS checksum mismatch: {path}')
        return {'path': entry['path'], 'bytes': path.stat().st_size, 'sha256': digest}
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, record in enumerate(pool.map(download, entries), 1):
            records.append(record)
            if i % 25 == 0:
                print(f'Downloaded/verified {i}/{len(entries)}', flush=True)
    manifest = {'source': REPO, 'revision': revision,
                'official_dataset': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
                'license': 'CC BY-NC-SA 4.0',
                'note': 'Mirror checksums verified where published; no independent official checksum comparison.',
                'files': records}
    (args.root / f'download_manifest_{args.category}.json').write_text(json.dumps(manifest, indent=2))
    print(f'Complete: {len(records)} files; revision {revision}', flush=True)

if __name__ == '__main__':
    main()
