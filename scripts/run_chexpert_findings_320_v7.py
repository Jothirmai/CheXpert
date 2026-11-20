"""Complete the 320-pixel extraction, bounded CUDA fit, and expert evaluation."""

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = 'ca5a840e438b5d2d0fa07f560782f196b38a84aa'
SOURCE = Path.home() / '.cache/huggingface/hub/datasets--danjacobellis--chexpert/snapshots' / REVISION / 'data'
DATA = ROOT / 'data/chexpert_plus_cnn_findings_320_v1'
OUT = ROOT / 'checkpoints/chexpert_plus_findings_320_v7'


def run(script, *args):
    subprocess.run([sys.executable, '-u', f'scripts/{script}', *args], cwd=ROOT, check=True)


def main():
    files = [SOURCE / f'train-{index:05d}-of-00023.parquet' for index in range(23)]
    files.append(SOURCE / 'validation-00000-of-00001.parquet')
    started = time.monotonic()
    while not all(path.is_file() for path in files):
        if time.monotonic() - started > 3 * 3600:
            raise RuntimeError('Pinned CheXpert image snapshot did not finish downloading')
        print('WAITING_FOR_SHARDS', sum(path.is_file() for path in files), '/', len(files), flush=True)
        time.sleep(60)
    if not (DATA / 'manifest.json').exists():
        run('prepare_chexpert_findings_320.py')
    print('DATA_READY', flush=True)
    if not (OUT / 'protocol.json').exists():
        run('train_chexpert_findings_320_v7.py', '--epochs', '2', '--batch-size', '12', '--workers', '8')
    if json.loads((OUT / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('320-pixel training is incomplete')
    if not (OUT / 'expert_valid_comparison.json').exists():
        run('evaluate_chexpert_findings_320_v7.py')
    comparison = json.loads((OUT / 'expert_valid_comparison.json').read_text())
    if comparison['candidate_320']['macro_auroc'] > comparison['source_224']['macro_auroc']:
        if not (OUT / 'test_500_exploratory.json').exists():
            run('evaluate_chexpert_test_500_exploratory_v7.py')
    else:
        print('SKIPPED_REPEAT_TEST: no expert-validation gain', flush=True)
    print('CHEXPERT_320_V7_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
