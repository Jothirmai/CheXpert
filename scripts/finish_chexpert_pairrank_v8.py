"""Evaluate the pairwise-ranking CNN and gate repeat test on development gains."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'checkpoints/chexpert_plus_pairrank_v8'
SOURCE = ROOT / 'checkpoints/chexpert_plus_densenet201_v6'


def run(script, *args):
    subprocess.run([sys.executable, '-u', f'scripts/{script}', *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    while time.monotonic() - started < 12 * 3600:
        try:
            if (OUT / 'best_model.pth').exists() and json.loads((OUT / 'progress.json').read_text()).get('phase') == 'complete':
                break
        except (OSError, ValueError):
            pass
        try:
            os.kill(args.training_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError('Training ended before a complete checkpoint was written') from exc
        time.sleep(30)
    else:
        raise RuntimeError('Pairwise-ranking training exceeded 12 hours')
    run('evaluate_chexpert_expert_valid.py', '--candidate', '--candidate-out',
        'checkpoints/chexpert_plus_pairrank_v8')
    run('bootstrap_chexpert_expert_valid.py', '--candidate-out',
        'checkpoints/chexpert_plus_pairrank_v8')
    run('evaluate_chexpert_development_five.py', '--checkpoint-dir',
        'checkpoints/chexpert_plus_pairrank_v8')
    candidate_dev = json.loads((OUT / 'impression_development_five.json').read_text())['macro_auroc']
    source_dev = json.loads((SOURCE / 'impression_development_five.json').read_text())['macro_auroc']
    expert = json.loads((OUT / 'expert_valid_comparison.json').read_text())['results']
    if candidate_dev > source_dev and expert['ours_pairrank_v8']['macro_auroc'] > expert['ours_densenet201_v6']['macro_auroc']:
        run('evaluate_chexpert_test_500_exploratory_v7.py', '--checkpoint-dir',
            'checkpoints/chexpert_plus_pairrank_v8', '--image-size', '224')
    else:
        print('SKIPPED_REPEAT_TEST: no joint development and expert-validation gain', flush=True)
    print('CHEXPERT_PAIRRANK_V8_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
