"""Wait for the frozen CNN run, then evaluate the selected checkpoint."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--training-pid', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / 'checkpoints/chexpert_plus_impression_v3'
started = time.monotonic()

while time.monotonic() - started < 12 * 3600:
    try:
        status = json.loads((out / 'progress.json').read_text())
        if status.get('phase') == 'complete' and (out / 'best_model.pth').exists():
            break
    except (OSError, ValueError):
        pass
    try:
        os.kill(args.training_pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError('Training stopped before completion; inspect the training log') from exc
    time.sleep(20)
else:
    raise RuntimeError('Training did not finish in twelve hours')

result = subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_plus_impression.py'], cwd=root)
if result.returncode:
    raise SystemExit(result.returncode)
expert = subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_expert_valid.py', '--candidate'], cwd=root)
if expert.returncode:
    raise SystemExit(expert.returncode)
expert_bootstrap = subprocess.run([sys.executable, '-u', 'scripts/bootstrap_chexpert_expert_valid.py'], cwd=root)
if expert_bootstrap.returncode:
    raise SystemExit(expert_bootstrap.returncode)
bootstrap = subprocess.run([sys.executable, '-u', 'scripts/bootstrap_chexpert_plus_impression.py'], cwd=root)
if bootstrap.returncode:
    raise SystemExit(bootstrap.returncode)
print('CHEXPERT_PLUS_IMPRESSION_COMPARISON_COMPLETE', flush=True)
