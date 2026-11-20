"""Queue the study-level fusion evaluation behind lateral-branch training."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_STATUS = ROOT / 'reports/chexpert_plus_lateral_320_v13_status.json'
STATUS = ROOT / 'reports/chexpert_multiview_v14_status.json'
TRAIN_PID = 324886


def update(phase, **details):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(payload, indent=2))
    print('STATUS', json.dumps(payload), flush=True)


def main():
    update('waiting_for_lateral_training')
    started = time.monotonic()
    while time.monotonic() - started < 12 * 3600:
        state = json.loads(TRAIN_STATUS.read_text())
        if state['phase'] == 'training_complete':
            break
        if state['phase'] == 'failed':
            raise RuntimeError('Lateral training failed')
        try:
            os.kill(TRAIN_PID, 0)
        except ProcessLookupError as error:
            raise RuntimeError('Lateral training ended without completion status') from error
        time.sleep(30)
    else:
        raise RuntimeError('Lateral training exceeded 12 hours')
    update('evaluating_study_fusion')
    subprocess.run([sys.executable, '-u', 'scripts/evaluate_chexpert_multiview_v14.py'],
                   cwd=ROOT, check=True)
    result = json.loads((ROOT / 'reports/chexpert_multiview_v14.json').read_text())
    exploratory = ROOT / 'reports/chexpert_multiview_v14_test_exploratory.json'
    if exploratory.exists():
        score = json.loads(exploratory.read_text())['macro_auroc']
        update('fresh_test_required' if score >= .93 else 'complete_below_target',
               selected=result['selected'], exploratory_test_macro_auroc=score)
    else:
        update('complete_no_joint_gain', selected=result['selected'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update('failed', error=repr(error))
        raise
