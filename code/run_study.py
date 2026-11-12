"""Execute the locked 3-category, 3-seed, 3-budget study, resumably."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT.parent/'mtr/data')
    parser.add_argument('--cache', type=Path, default=ROOT/'cache')
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/study.json').read_text())
    args.results.mkdir(parents=True,exist_ok=True)
    lock = args.results/'study_locked.json'
    identity = {'config':config, 'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [ROOT/'code/core.py',ROOT/'code/run_experiment.py']}}
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise ValueError('Study code/config changed after lock; preserve old results and start a new study')
    lock.write_text(json.dumps(identity, indent=2))
    for category in config['categories']:
        subprocess.run([sys.executable,str(ROOT/'code/download_data.py'),'--root',str(args.data),
                        '--category',category,'--revision',config['dataset_revision']],check=True)
    for budget in config['budgets']:
        for category in config['categories']:
            for seed in config['seeds']:
                name = f'{category}_s{seed}_b{budget}'
                out = args.results/name
                if (out/'completion.json').exists():
                    print('Already complete:',name,flush=True); continue
                out.mkdir(parents=True,exist_ok=True)
                run_config = config | {'category':category,'seed':seed,'budget':budget}
                path = ROOT/'configs'/f'{name}.json'
                path.write_text(json.dumps(run_config,indent=2))
                print('Starting:',name,datetime.now(timezone.utc).isoformat(),flush=True)
                with (out/'run.log').open('a', buffering=1) as log:
                    subprocess.run([sys.executable,'-u',str(ROOT/'code/run_experiment.py'),
                        '--data',str(args.data),'--output',str(out),'--cache',str(args.cache/f'{category}_s{seed}'),
                        '--feature-cache',str(args.cache/'test'/category),'--config',str(path)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                print('Completed:',name,flush=True)
    (args.results/'study_completion.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
                    'experiments':27,'methods_per_experiment':6,'conditions_per_experiment':10},indent=2))

if __name__ == '__main__':
    main()
