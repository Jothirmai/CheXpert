"""Train a lateral-view branch after its frozen dataset is prepared."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/chexpert_plus_cnn_lateral_320_v1'
OUT = ROOT / 'checkpoints/chexpert_plus_lateral_320_v13'
STATUS = ROOT / 'reports/chexpert_plus_lateral_320_v13_status.json'
PREPARE_PID = 324833


def update(phase, **details):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(payload, indent=2))
    print('STATUS', json.dumps(payload), flush=True)


def main():
    update('waiting_for_lateral_images')
    start = time.monotonic()
    while not (DATA / 'manifest.json').is_file():
        if time.monotonic() - start > 3600:
            raise RuntimeError('Lateral extraction exceeded one hour')
        try:
            os.kill(PREPARE_PID, 0)
        except ProcessLookupError as error:
            raise RuntimeError('Lateral extraction ended without a manifest') from error
        time.sleep(30)
    counts = json.loads((DATA / 'manifest.json').read_text())['counts']
    update('training', fit_images=counts['fit']['images'], development_images=counts['development']['images'])
    subprocess.run([
        sys.executable, '-u', 'scripts/train_chexpert_findings_320_v7.py',
        '--data-root', str(DATA.relative_to(ROOT)),
        '--output', str(OUT.relative_to(ROOT)),
        '--source', 'checkpoints/chexpert_plus_impression_320_v12/best_model.pth',
        '--learning-rate', '1e-5', '--seed', '48',
        '--epochs', '2', '--batch-size', '12', '--workers', '8',
    ], cwd=ROOT, check=True)
    best = json.loads((OUT / 'best_metrics.json').read_text())
    update('training_complete', best_epoch=best['epoch'],
           lateral_development_five_auroc=best['five_finding_macro_auroc'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update('failed', error=repr(error))
        raise
