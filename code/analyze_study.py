"""Aggregate measured results; paired image bootstrap with seeds averaged first."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'clean':'Clean-only','pooled':'Pooled','uniform':'Uniform + guard','adaptive':'Adaptive + guard',
         'no_guard':'Without guard','no_anchor':'Without anchors'}
METRICS = ['image_auroc','pixel_auroc','pixel_ap','fpr','recall']

def save_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary-only',action='store_true',help='Preview the completed B4096 study separately while secondary budgets run')
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/study.json').read_text())
    if args.primary_only:config['budgets']=[4096]
    result=args.results; analysis=result/('primary_preview' if args.primary_only else 'analysis'); analysis.mkdir(exist_ok=True)
    figures=analysis/'figures' if args.primary_only else ROOT/'report/figures';figures.mkdir(parents=True,exist_ok=True)
    rows=[];scores=[];decisions=[];efficiency=[]
    for budget in config['budgets']:
        for category in config['categories']:
            for seed in config['seeds']:
                folder=result/f'{category}_s{seed}_b{budget}'
                if not (folder/'completion.json').exists():
                    raise RuntimeError(f'Incomplete experiment: {folder.name}')
                identity={'budget':budget,'category':category,'seed':seed}
                with (folder/'metrics.csv').open() as f:
                    rows.extend(identity|{k:float(v) if k in METRICS else v for k,v in r.items()} for r in csv.DictReader(f))
                with (folder/'per_image_scores.csv').open() as f:
                    scores.extend(identity|{k:float(v) if k=='score' else int(v) if k in ['label','prediction'] else v for k,v in r.items()} for r in csv.DictReader(f))
                fit=json.loads((folder/'fit_summary.json').read_text())
                latency=json.loads((folder/'latency.json').read_text())
                for m in config['methods']:
                    efficiency.append(identity|{'method':m,'bank_bytes':fit['descriptor_bank_bytes'],
                        'construction_seconds':fit['bank_construction_seconds'][m],
                        'latency_ms':latency[m]['median_ms'],
                        'peak_cuda_bytes':fit['peak_cuda_allocated_bytes_through_fitting']})
                for m,d in fit['choices'].items():
                    decisions.append(identity|{'method':m,'fallback':d['fallback'],'rho':d.get('rho'),
                       'clean_allocation':d['allocation'][0],'brightness_allocation':d['allocation'][1],
                       'jpeg_allocation':d['allocation'][2]})
    save_csv(analysis/'all_metrics.csv',rows);save_csv(analysis/'all_scores.csv',scores)
    failures=[r for r in scores if r['prediction']!=r['label']]
    if failures:save_csv(analysis/'failure_cases.csv',failures)
    save_csv(analysis/'efficiency.csv',efficiency);save_csv(analysis/'decisions.csv',decisions)
    summary=[];category_rows=[];seed_rows=[]
    for b in config['budgets']:
        for m in config['methods']:
            for scope in ['clean','corrupted']:
                selected=[r for r in rows if r['budget']==b and r['method']==m and (r['condition']=='clean')==(scope=='clean')]
                summary.append({'budget':b,'method':m,'scope':scope}|{k:float(np.mean([r[k] for r in selected])) for k in METRICS})
                for cat in config['categories']:
                    sub=[r for r in selected if r['category']==cat]
                    category_rows.append({'budget':b,'method':m,'scope':scope,'category':cat}|{k:float(np.mean([r[k] for r in sub])) for k in METRICS})
                for seed in config['seeds']:
                    sub=[r for r in selected if r['seed']==seed]
                    seed_rows.append({'budget':b,'method':m,'scope':scope,'seed':seed}|{k:float(np.mean([r[k] for r in sub])) for k in METRICS})
    save_csv(analysis/'summary.csv',summary);save_csv(analysis/'by_category.csv',category_rows);save_csv(analysis/'by_seed.csv',seed_rows)
    worst=[]
    for b in config['budgets']:
        for cat in config['categories']:
            for seed in config['seeds']:
                for m in config['methods']:
                    r=min((r for r in rows if r['budget']==b and r['category']==cat and r['seed']==seed and r['method']==m),key=lambda r:r['image_auroc'])
                    worst.append(r)
    save_csv(analysis/'worst_condition.csv',worst)
    families=[]
    for b in config['budgets']:
        for m in config['methods']:
            for family in ['brightness','jpeg','blur']:
                sub=[r for r in rows if r['budget']==b and r['method']==m and r['condition'].startswith(family)]
                families.append({'budget':b,'method':m,'family':family}|{k:float(np.mean([r[k] for r in sub])) for k in METRICS})
    save_csv(analysis/'by_corruption_family.csv',families)
    primary=[r for r in scores if r['budget']==4096 and r['method'] in ['adaptive','pooled'] and r['condition']!='clean']
    rng=np.random.default_rng(config['bootstrap_seed']);boot=np.zeros(config['bootstrap_resamples']);point=0.;cat_stats=[]
    for cat in config['categories']:
        values=[r for r in primary if r['category']==cat]
        files=sorted({r['file'] for r in values});views=config['test_views'][1:]
        index={(r['method'],r['seed'],r['condition'],r['file']):r['score'] for r in values}
        labels=np.array([next(r['label'] for r in values if r['file']==f) for f in files])
        arrays={m:np.array([[np.mean([index[m,s,v,f] for s in config['seeds']]) for f in files] for v in views]) for m in ['adaptive','pooled']}
        wins={}
        for m,a in arrays.items():
            difference=a[:,labels==1,None]-a[:,None,labels==0]
            wins[m]=np.mean((difference>0)+.5*(difference==0),axis=0)
        delta=wins['adaptive']-wins['pooled'];point+=delta.mean()/len(config['categories'])
        p,n=delta.shape
        # Counts resample original images, shared over all views and both methods.
        pos=rng.multinomial(p,np.full(p,1/p),size=len(boot))
        neg=rng.multinomial(n,np.full(n,1/n),size=len(boot))
        boot+=np.einsum('bi,ij,bj->b',pos,delta,neg)/(p*n*len(config['categories']))
        cat_stats.append({'category':cat,'adaptive':float(wins['adaptive'].mean()),'pooled':float(wins['pooled'].mean()),'delta':float(delta.mean())})
    ci=np.quantile(boot,[.025,.975])
    clean={m:next(r['image_auroc'] for r in summary if r['budget']==4096 and r['method']==m and r['scope']=='clean') for m in ['adaptive','clean']}
    inference={'comparison':'Adaptive minus pooled; B=4096; mean of 9 corruption AUROCs, macro over 3 categories; per-image scores averaged over 3 seeds before AUROC',
        'delta':float(point),'ci95':ci.tolist(),'resamples':len(boot),'seed':config['bootstrap_seed'],
        'categories':cat_stats,'clean_macro_seed_mean_delta_vs_clean':clean['adaptive']-clean['clean'],
        'success_criterion_met':bool(ci[0]>0 and clean['adaptive']-clean['clean']>=-.01),
        'limitations':'Conditional image-sampling uncertainty for these 3 categories and fitted seeds; does not bootstrap new categories or training sets.'}
    (analysis/'primary_test.json').write_text(json.dumps(inference,indent=2))
    np.save(analysis/'bootstrap_deltas.npy',boot)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for m in config['methods'][:4]:
        selected=[next(r for r in summary if r['budget']==b and r['method']==m and r['scope']=='corrupted') for b in sorted(config['budgets'])]
        for ax,key in zip(axes,['image_auroc','fpr']):ax.plot(sorted(config['budgets']),[100*r[key] for r in selected],marker='o',label=NAMES[m])
    for ax,title in zip(axes,['Mean corrupted image AUROC (%)','Mean corrupted FPR (%)']):
        ax.set(xlabel='Memory vectors',ylabel=title)
        ax.set_xticks(sorted(config['budgets']),[f'{b:,}' for b in sorted(config['budgets'])]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=11);fig.tight_layout();fig.savefig(figures/'budget_tradeoff.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));x=np.arange(10)
    for m in config['methods'][:4]:
        for ax,k in zip(axes,['image_auroc','fpr']):
            ax.plot(x,[100*np.mean([r[k] for r in rows if r['budget']==4096 and r['method']==m and r['condition']==v]) for v in config['test_views']],marker='o',label=NAMES[m])
    for ax,label in zip(axes,['Image AUROC (%)','False-positive rate (%)']):
        ax.set_xticks(x,['Clean','B.75','B.60','B.45','J70','J50','J30','G.5','G1','G1.5'],rotation=45);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=11);fig.tight_layout();fig.savefig(figures/'conditions.png',dpi=200);plt.close(fig)
    if any(not list((result/f'{cat}_s23_b4096/examples').glob('*_adaptive.npy')) for cat in config['categories']):
        print('Numeric analysis completed. Raw example maps are absent; the supplied qualitative figure is retained. Regenerate banks/maps with run_study.py to rebuild it.')
        print(json.dumps(inference,indent=2))
        return
    from PIL import Image
    fig=plt.figure(figsize=(8,10),layout='constrained')
    grid=fig.add_gridspec(6,5,width_ratios=[1,1,1,1,.06])
    axes=np.array([[fig.add_subplot(grid[r,k]) for k in range(4)] for r in range(6)])
    for c,cat in enumerate(config['categories']):
        folder=result/f'{cat}_s23_b4096';split=json.loads((folder/'split_manifest.json').read_text())
        defect=next(Path(p).parent.name for p in split['test'] if Path(p).parent.name!='good')
        for j,(label,view) in enumerate([(defect,'clean'),('good','brightness_0.45')]):
            key=f'{label}_{view}';r=c*2+j;example=folder/'examples'
            im=np.asarray(Image.open(example/f'{key}.png'));mask=np.asarray(Image.open(example/f'{key}_mask.png'))
            maps=[np.load(example/f'{key}_{m}.npy') for m in ['clean','adaptive']]
            low=min(x.min() for x in maps);high=max(x.max() for x in maps)
            label_text='Normal' if label=='good' else label.replace('_',' ').title()
            view_text='Clean' if view=='clean' else 'Brightness 0.45'
            axes[r,0].imshow(im);axes[r,0].set_ylabel(f'{cat.title()}\n{label_text}\n{view_text}',fontsize=11)
            axes[r,1].imshow(mask,cmap='gray',vmin=0,vmax=255)
            for k,heat in enumerate(maps,2):
                axes[r,k].imshow(im);colored=axes[r,k].imshow(heat,cmap='inferno',alpha=.55,vmin=low,vmax=high)
            bar=fig.colorbar(colored,cax=fig.add_subplot(grid[r,4]),ticks=[low,high],format='%.2f')
            bar.ax.tick_params(labelsize=10)
            for ax in axes[r]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Input','Defect mask','Clean-only','Adaptive']):ax.set_title(title,fontsize=11)
    fig.savefig(figures/'qualitative.png',dpi=200);plt.close(fig)
    print(json.dumps(inference,indent=2))

if __name__=='__main__':main()
