"""Finish analysis, audit and documents once every experiment is complete."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'mtr/data')
    parser.add_argument('--tectonic',type=Path,default=ROOT.parent/'tmp/tools/tectonic/tectonic.exe')
    args=parser.parse_args()
    while not (ROOT/'results/study_completion.json').exists():
        if not args.wait:raise RuntimeError('Study is incomplete')
        time.sleep(20)
    commands=[['code/analyze_study.py'],['code/audit_study.py','--data',str(args.data),'--pixels'],['report/build_documents.py']]
    for command in commands:
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,check=True)
    subprocess.run([str(args.tectonic),'--keep-logs','main.tex'],cwd=ROOT/'paper',check=True)
    shutil.copy2(ROOT/'paper/main.pdf',ROOT/'group23-final-paper.pdf')
    print('Documents built; visual inspection is required before packaging/publication.',flush=True)

if __name__=='__main__':main()
