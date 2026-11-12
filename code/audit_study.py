"""Audit all image metrics/protocol invariants and independent sampled pixel metrics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT.parent/'mtr/data')
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--pixels',action='store_true',help='Independently recompute pixels for clean+strong brightness, primary seed 23, all categories/methods')
    parser.add_argument('--run',default=None,help='Optional single run name for an early partial audit')
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/study.json').read_text());count=0;pixel_checks=0;experiments=0
    locked=json.loads((args.results/'study_locked.json').read_text())
    assert locked['config']==cfg
    for name,digest in locked['code_sha256'].items():assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==digest
    for b in cfg['budgets']:
        for cat in cfg['categories']:
            for seed in cfg['seeds']:
                folder=args.results/f'{cat}_s{seed}_b{b}'
                if args.run and folder.name!=args.run:continue
                experiments+=1
                done=json.loads((folder/'completion.json').read_text())
                split=json.loads((folder/'split_manifest.json').read_text())
                sets=[set(v) for v in split.values()]
                assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
                fit=json.loads((folder/'fit_summary.json').read_text());cal=json.loads((folder/'calibration_scores.json').read_text())
                limit=cfg['clean_tolerance']*max(fit['reference_clean_coverage'],1e-12)
                for method in ['uniform','adaptive']:
                    choice=fit['choices'][method]
                    eligible=[o for o in choice['options'] if o['coverage'][0]<=limit]
                    assert choice['fallback']==(not eligible)
                    if eligible:
                        expected=min(eligible,key=lambda o:(max(o['coverage'][1:]),-o['rho']))
                        for key in ['rho','coverage','allocation']:assert choice[key]==expected[key]
                expected=min(fit['choices']['adaptive']['options'],key=lambda o:(max(o['coverage'][1:]),-o['rho']))
                for key in ['rho','coverage','allocation']:assert fit['choices']['no_guard'][key]==expected[key]
                assert fit['choices']['no_anchor']['fallback']==(fit['choices']['no_anchor']['coverage'][0]>limit)
                for choice in fit['choices'].values():assert sum(choice['allocation'])==b
                rows=list(csv.DictReader((folder/'metrics.csv').open()));scores=list(csv.DictReader((folder/'per_image_scores.csv').open()))
                assert len(rows)==60 and len(scores)==len(split['test'])*60
                assert done['metrics_rows']==len(rows) and done['score_rows']==len(scores)
                banks={m:np.load(folder/f'bank_{m}.npy') for m in cfg['methods']}
                for m,bank in banks.items():
                    assert bank.shape==(b,1024) and bank.dtype==np.float32 and np.isfinite(bank).all()
                    assert len(np.unique(bank,axis=0))==b
                    rank=math.ceil(.95*(len(cal[m])+1));threshold=sorted(cal[m])[rank-1] if rank<=len(cal[m]) else None
                    assert threshold==fit['thresholds'][m]['value']
                    if m in ['uniform','adaptive','no_guard']:
                        choice=fit['choices'][m]
                        anchors=b if choice['fallback'] else math.floor(choice['rho']*b)
                        assert np.array_equal(bank[:anchors],banks['clean'][:anchors])
                    if m in fit['choices'] and fit['choices'][m]['fallback']:assert np.array_equal(bank,banks['clean'])
                for row in rows:
                    sub=[r for r in scores if r['method']==row['method'] and r['condition']==row['condition']]
                    assert [r['file'] for r in sub]==split['test']
                    y=np.array([int(r['label']) for r in sub]);s=np.array([float(r['score']) for r in sub]);pred=np.array([int(r['prediction']) for r in sub])
                    threshold=fit['thresholds'][row['method']]['value']
                    assert np.array_equal(pred,s>(threshold if threshold is not None else np.inf))
                    assert np.array_equal(y,[Path(r['file']).parent.name!='good' for r in sub])
                    for k,v in {'image_auroc':roc_auc_score(y,s),'fpr':pred[y==0].mean(),'recall':pred[y==1].mean()}.items():
                        assert abs(float(row[k])-v)<1e-10,(folder.name,row,k,v)
                    count+=1
                for view in cfg['test_views']:
                    with np.load(folder/f'patch_scores_{view}.npz') as grids:
                        for m in cfg['methods']:
                            assert grids[m].shape==(len(split['test']),32,32) and np.isfinite(grids[m]).all()
                        if args.pixels and b==4096 and seed==23 and view in ['clean','brightness_0.45']:
                            masks=[]
                            for name in split['test']:
                                p=Path(name)
                                if p.parent.name=='good':masks.append(np.zeros((256,256),dtype=bool))
                                else:
                                    with Image.open(args.data/cat/'ground_truth'/p.parent.name/(p.stem+'_mask.png')) as im:
                                        masks.append(np.array(im.resize((256,256),Image.Resampling.NEAREST))>0)
                            truth=np.stack(masks).ravel()
                            for m in cfg['methods']:
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                                maps=F.interpolate(torch.from_numpy(grids[m]).to(device)[:,None],size=(256,256),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
                                pixels=np.stack([gaussian_filter(x,sigma=4.) for x in maps]).ravel()
                                row=next(r for r in rows if r['method']==m and r['condition']==view)
                                tolerance=1e-10 if device=='cuda' else 3e-6
                                assert abs(roc_auc_score(truth,pixels)-float(row['pixel_auroc']))<tolerance
                                assert abs(average_precision_score(truth,pixels)-float(row['pixel_ap']))<tolerance
                                pixel_checks+=1
                print('Audited',folder.name,flush=True)
    assert experiments==(1 if args.run else 27)
    report={'passed':True,'experiments':experiments,'image_metric_rows_checked':count,'pixel_metric_rows_independently_checked':pixel_checks,
            'pixel_scope':('B4096, seed23, clean and brightness0.45, all methods; '+('run '+args.run if args.run else 'all categories')) if args.pixels else 'Not requested',
            'all_patch_grids_checked':True,'split_threshold_bank_anchor_checks':True,'saved_selection_decisions_checked':True,'locked_code_hashes_verified':True}
    name=f'audit_{args.run}.json' if args.run else 'audit.json'
    (args.results/name).write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
