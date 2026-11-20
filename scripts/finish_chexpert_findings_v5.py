"""Evaluate the findings-section CNN after its bounded fit completes."""
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
out = root / 'checkpoints/chexpert_plus_findings_v5'
started = time.monotonic()
while time.monotonic() - started < 8 * 3600:
    try:
        state = json.loads((out / 'progress.json').read_text())
        if state.get('phase') == 'complete' and (out / 'best_model.pth').exists():
            break
    except (OSError, ValueError):
        pass
    try:
        os.kill(args.training_pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError('Training ended without a complete checkpoint') from exc
    time.sleep(20)
else:
    raise RuntimeError('Training exceeded the eight-hour limit')
command = [sys.executable, '-u', 'scripts/evaluate_chexpert_expert_valid.py', '--candidate', '--candidate-out', 'checkpoints/chexpert_plus_findings_v5']
result = subprocess.run(command, cwd=root)
if result.returncode:
    raise SystemExit(result.returncode)
bootstrap = subprocess.run([sys.executable, '-u', 'scripts/bootstrap_chexpert_expert_valid.py', '--candidate-out', 'checkpoints/chexpert_plus_findings_v5'], cwd=root)
if bootstrap.returncode:
    raise SystemExit(bootstrap.returncode)
print('CHEXPERT_PLUS_FINDINGS_V5_EVALUATION_COMPLETE', flush=True)
