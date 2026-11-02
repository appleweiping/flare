"""Replay the independent audits against relocated source artifacts.

Original audit script bytes remain unchanged beside this wrapper. Copies run
inside a new output directory; this never runs classifier training or a study grid.
"""
import argparse
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--output-root',type=Path,required=True)
args=parser.parse_args()
args.output_root.mkdir(parents=True,exist_ok=False)
scripts=Path(__file__).resolve().parent
names=['reconstruct_block1.py','reconstruct_block2.py','replay_saved_predictions.py']
for name in names:
    source=(scripts/name).read_text()
    source=source.replace("Path('/workspace/.research/flare-uq')",f'Path({str(args.data_root.resolve())!r})')
    (args.output_root/name).write_text(source)
for name in names:
    command=[sys.executable,str(args.output_root/name)]
    if name=='reconstruct_block2.py':command.append(str(args.data_root/'allocation-v1'))
    result=subprocess.run(command,check=False,capture_output=True,text=True)
    (args.output_root/(name+'.stdout.txt')).write_text(result.stdout)
    (args.output_root/(name+'.stderr.txt')).write_text(result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    print(name+' completed')
