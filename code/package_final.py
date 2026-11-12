"""Package the verified final documents/source, with checksums; omit caches/data."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[1]
def main():
    audit=json.loads((ROOT/'results/audit.json').read_text());assert audit['passed']
    for name in ['group23-final-paper.pdf','group23-final-report.pdf']:
        if not (ROOT/name).exists():raise FileNotFoundError(name)
    primary=json.loads((ROOT/'results/analysis/primary_test.json').read_text())
    citation=ROOT/'CITATION.cff'
    release_date=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5,minutes=30))).date().isoformat()
    citation.write_text('\n'.join('date-released: '+release_date if line.startswith('date-released:') else line
                                  for line in citation.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    with (ROOT/'results/analysis/summary.csv').open() as f:summary=list(csv.DictReader(f))
    out=ROOT.parent/'output';(out/'pdf').mkdir(parents=True,exist_ok=True)
    for name in ['group23-final-paper.pdf','group23-final-report.pdf']:shutil.copy2(ROOT/name,out/'pdf'/name)
    index=out/'README.md';backup=out/'README-preliminary.md'
    if index.exists() and not backup.exists():shutil.copy2(index,backup)
    index.write_text('# Final project deliverables\n\n'
        '- [Final CVPR-format paper](pdf/group23-final-paper.pdf)\n'
        '- [Final institutional report](pdf/group23-final-report.pdf)\n'
        '- [Complete source and results package](group23-final-project.zip)\n'
        '- [Overleaf paper source](group23-final-paper-overleaf.zip)\n\n'
        'The completed working project is in `../final/`. Earlier proposal and midterm artifacts remain preserved separately. '
        'Historical preliminary-report instructions are in [README-preliminary.md](README-preliminary.md).\n',encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    start=text.index('The final study is currently executing.') if 'The final study is currently executing.' in text else text.index('**Completed and audited:**')
    end=text.index('## Experiment design')
    table=['| Method | Corrupted image AUROC | Pixel AP | FPR |','|---|---:|---:|---:|']
    for m in ['clean','pooled','uniform','adaptive','no_guard','no_anchor']:
        r=next(r for r in summary if int(r['budget'])==4096 and r['method']==m and r['scope']=='corrupted')
        table.append('| '+m+' | '+' | '.join(f'{100*float(r[k]):.2f}%' for k in ['image_auroc','pixel_ap','fpr'])+' |')
    p=primary
    status=('**Completed and audited:** 27 category/seed/budget experiments, six methods and ten conditions each (1,620 metric rows).\n\n'
        '[Final paper](group23-final-paper.pdf) · [Final report](group23-final-report.pdf) · [Full results](results/analysis/summary.csv)\n\n'
        'At B=4,096, metrics below equally average categories, seeds and nine corrupted conditions.\n\n'+'\n'.join(table)+'\n\n'
        f'The primary adaptive-minus-pooled comparison using seed-averaged scores is **{100*p["delta"]:+.2f} percentage points**, '
        f'95% paired bootstrap interval **[{100*p["ci95"][0]:+.2f}, {100*p["ci95"][1]:+.2f}]**. '
        f'The prespecified success criterion was **{"met" if p["success_criterion_met"] else "not met"}**. '
        'The table averages per-seed AUROCs; the primary test averages scores before computing AUROC. '
        'These different aggregation orders can produce differences with opposite signs. '
        'Negative findings are retained. The earlier midterm pilot is not pooled with the final study.\n\n'
        '![Budget trade-off](report/figures/budget_tradeoff.png)\n\n')
    readme.write_text(text[:start]+status+text[end:],encoding='utf-8')
    def publishable(p):
        rel=p.relative_to(ROOT)
        return not any(x in ['cache','reproduced_cache','reproduced_results','primary_preview','prediction','data','tmp','__pycache__','.git'] for x in rel.parts) and p.suffix not in ['.pyc','.log','.aux','.out','.bbl','.blg','.zip'] and not (rel.parts[0]=='results' and p.name.startswith(('bank_','patch_scores_'))) and not ('examples' in rel.parts and p.suffix=='.npy') and p.name!='SHA256SUMS.json'
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and publishable(p))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    files.append(ROOT/'SHA256SUMS.json')
    with zipfile.ZipFile(out/'group23-final-project.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'group23-final-project/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'group23-final-paper-overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'paper').glob('*')):
            if p.is_file() and p.suffix in ['.tex','.sty','.bst','.bib']:
                if p.name=='main.tex':
                    z.writestr('main.tex',p.read_text(encoding='utf-8').replace('../report/figures/','figures/'))
                else:z.write(p,p.name)
        for p in sorted((ROOT/'report/figures').glob('*.png')):z.write(p,'figures/'+p.name)
        z.writestr('README.txt','Compile the root main.tex with pdfLaTeX. Figures are in figures/. The bibliography and supplied CVPR style are included at the project root.\n')
    print('Packaged',len(files),'files; full fitted banks and patch grids remain locally in final/results.')

if __name__=='__main__':main()
