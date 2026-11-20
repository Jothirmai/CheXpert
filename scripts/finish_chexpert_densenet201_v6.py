"""Evaluate the DenseNet201 only after its bounded CUDA run completes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'checkpoints/chexpert_plus_densenet201_v6'
    started = time.monotonic()
    while time.monotonic() - started < 16 * 3600:
        try:
            progress = json.loads((out / 'progress.json').read_text())
            if progress.get('phase') == 'complete' and (out / 'best_model.pth').exists():
                break
        except (OSError, ValueError):
            pass
        try:
            os.kill(args.training_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError('Training ended before a complete checkpoint was written') from exc
        time.sleep(30)
    else:
        raise RuntimeError('Training exceeded the 16-hour limit')
    for script in ('evaluate_chexpert_expert_valid.py', 'bootstrap_chexpert_expert_valid.py'):
        command = [sys.executable, '-u', f'scripts/{script}', '--candidate-out',
                   'checkpoints/chexpert_plus_densenet201_v6']
        if script == 'evaluate_chexpert_expert_valid.py':
            command.append('--candidate')
        subprocess.run(command, cwd=root, check=True)
    subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_development_five.py',
                    '--checkpoint-dir', 'checkpoints/chexpert_plus_densenet201_v6'],
                   cwd=root, check=True)
    if not (root / 'checkpoints/chexpert_plus_findings_v5/impression_development_five.json').exists():
        subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_development_five.py',
                        '--checkpoint-dir', 'checkpoints/chexpert_plus_findings_v5'],
                       cwd=root, check=True)
    subprocess.run([sys.executable, '-u', 'scripts/select_chexpert_test_model.py'],
                   cwd=root, check=True)
    if (root / 'data/chexpert_test_500/image_manifest.json').exists():
        subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_test_500.py'],
                       cwd=root, check=True)
    print('CHEXPERT_PLUS_DENSENET201_V6_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
