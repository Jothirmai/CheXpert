"""Queue category-wise head tuning and full-split EfficientNetV2-S training."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / 'checkpoints/chexpert_plus_category_heads_v15'
EFFICIENT = ROOT / 'checkpoints/chexpert_plus_efficientnet_full_320_v16'
STATUS = ROOT / 'reports/chexpert_methods_v15_v16_status.json'
FEATURE_PID = 326819


def update(phase, **details):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(payload, indent=2))
    print('STATUS', json.dumps(payload), flush=True)


def run(script, *arguments):
    subprocess.run([sys.executable, '-u', f'scripts/{script}', *arguments], cwd=ROOT, check=True)


def wait_for_features():
    start = time.monotonic()
    while time.monotonic() - start < 12 * 3600:
        progress = HEAD / 'feature_progress.json'
        if progress.exists() and json.loads(progress.read_text()).get('phase') == 'complete':
            return
        try:
            os.kill(FEATURE_PID, 0)
        except ProcessLookupError as error:
            raise RuntimeError('Feature extraction ended without completion') from error
        time.sleep(30)
    raise RuntimeError('Feature extraction exceeded 12 hours')


def main():
    update('waiting_for_v12_features')
    wait_for_features()
    update('category_head_fitting')
    run('fit_chexpert_category_heads_v15.py')
    head_result = json.loads((HEAD / 'selection.json').read_text())
    head_dev = head_result['development_candidate']['macro_auroc']
    head_base_dev = head_result['development_baseline']['macro_auroc']
    head_expert = head_result['expert_validation_candidate_exploratory']['macro_auroc']
    head_base_expert = head_result['expert_validation_baseline_exploratory']['macro_auroc']
    head_details = {'head_development_auroc': head_dev,
                    'head_source_development_auroc': head_base_dev,
                    'head_expert_validation_auroc_exploratory': head_expert,
                    'head_source_expert_validation_auroc': head_base_expert}
    if head_dev > head_base_dev and head_expert > head_base_expert:
        update('category_head_exploratory_test', **head_details)
        run('evaluate_chexpert_test_500_exploratory_v7.py', '--checkpoint-dir',
            str(HEAD.relative_to(ROOT)), '--image-size', '320')
        head_test = json.loads((HEAD / 'test_500_exploratory.json').read_text())
        head_details['head_exploratory_test_auroc'] = head_test['macro_auroc']
    else:
        head_details['head_repeat_test_skipped'] = 'No joint development and expert-validation gain'
    update('full_efficientnet_training', **head_details)
    run('train_chexpert_alt_backbone_320.py', '--architecture', 'efficientnet_v2_s',
        '--data-root', 'data/chexpert_plus_cnn_impression_320_v1',
        '--output', str(EFFICIENT.relative_to(ROOT)), '--learning-rate', '5e-5',
        '--seed', '51', '--epochs', '3', '--batch-size', '12', '--workers', '8')
    update('full_efficientnet_expert_validation', **head_details)
    run('evaluate_chexpert_alt_backbone_320.py', '--checkpoint-dir',
        str(EFFICIENT.relative_to(ROOT)), '--data-root',
        'data/chexpert_plus_cnn_impression_320_v1', '--baseline-checkpoint-dir',
        'checkpoints/chexpert_plus_impression_320_v12')
    candidate_dev = json.loads((EFFICIENT / 'best_metrics.json').read_text())['five_finding_macro_auroc']
    baseline_dev = json.loads((ROOT / 'checkpoints/chexpert_plus_impression_320_v12/best_metrics.json').read_text())['five_finding_macro_auroc']
    comparison = json.loads((EFFICIENT / 'expert_valid_comparison.json').read_text())
    candidate_expert = comparison['candidate']['macro_auroc']
    baseline_expert = comparison['baseline']['macro_auroc']
    efficient_details = {'efficientnet_development_auroc': candidate_dev,
                         'densenet_development_auroc': baseline_dev,
                         'efficientnet_expert_validation_auroc_exploratory': candidate_expert,
                         'densenet_expert_validation_auroc': baseline_expert}
    if candidate_dev > baseline_dev and candidate_expert > baseline_expert:
        update('full_efficientnet_exploratory_test', **head_details, **efficient_details)
        run('evaluate_chexpert_test_500_exploratory_v7.py', '--checkpoint-dir',
            str(EFFICIENT.relative_to(ROOT)), '--image-size', '320')
        efficient_details['efficientnet_exploratory_test_auroc'] = json.loads(
            (EFFICIENT / 'test_500_exploratory.json').read_text())['macro_auroc']
    else:
        efficient_details['efficientnet_repeat_test_skipped'] = 'No joint development and expert-validation gain'
    scores = [head_details.get('head_exploratory_test_auroc', 0),
              efficient_details.get('efficientnet_exploratory_test_auroc', 0)]
    update('fresh_test_required' if max(scores) >= .93 else 'complete_below_target',
           **head_details, **efficient_details)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update('failed', error=repr(error))
        raise
