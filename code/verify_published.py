"""Verify published checksums and image metrics without data, banks or a GPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--skip-checksums',action='store_true',help='Use only for locally regenerated/modified artifacts')
    args=parser.parse_args()
    if not args.skip_checksums:
        hashes=json.loads((ROOT/'SHA256SUMS.json').read_text())
        for name,expected in hashes.items():
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Published artifact differs or is missing: {name}')
        print(f'Verified {len(hashes)} artifact checksums',flush=True)
    config=json.loads((ROOT/'configs/study.json').read_text());checked=0
    for budget in config['budgets']:
        for category in config['categories']:
            for seed in config['seeds']:
                run=args.results/f'{category}_s{seed}_b{budget}'
                split=json.loads((run/'split_manifest.json').read_text())
                fit=json.loads((run/'fit_summary.json').read_text())
                calibration=json.loads((run/'calibration_scores.json').read_text())
                metrics=list(csv.DictReader((run/'metrics.csv').open()))
                scores=list(csv.DictReader((run/'per_image_scores.csv').open()))
                assert len(metrics)==60 and len(scores)==60*len(split['test'])
                for method in config['methods']:
                    n=len(calibration[method]);rank=math.ceil(.95*(n+1))
                    threshold=sorted(calibration[method])[rank-1] if rank<=n else None
                    assert threshold==fit['thresholds'][method]['value']
                    for condition in config['test_views']:
                        values=[r for r in scores if r['method']==method and r['condition']==condition]
                        assert [r['file'] for r in values]==split['test']
                        y=np.array([int(r['label']) for r in values]);s=np.array([float(r['score']) for r in values])
                        pred=s>(threshold if threshold is not None else np.inf)
                        assert np.array_equal(y,[Path(r['file']).parent.name!='good' for r in values])
                        assert np.array_equal(pred,[int(r['prediction']) for r in values])
                        stored=next(r for r in metrics if r['method']==method and r['condition']==condition)
                        actual={'image_auroc':roc_auc_score(y,s),'fpr':pred[y==0].mean(),'recall':pred[y==1].mean()}
                        for key,value in actual.items():assert abs(float(stored[key])-value)<1e-10,(run.name,method,condition,key)
                        checked+=1
    print(f'PASS: {checked} image-metric rows across 27 experiments; pixel metrics and bank contents require the full audit.')

if __name__=='__main__':main()
