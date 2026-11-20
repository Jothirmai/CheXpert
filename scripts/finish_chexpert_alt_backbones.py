"""Queue two architecture comparisons behind the active CUDA training job.

The 500-study CheXpert challenge labels have already been inspected. Any repeat
score on that cohort is exploratory. A new, untouched cohort is required for a
confirmatory score if an exploratory run reaches the target.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/chexpert_alt_backbones_status.json'
PAIR_LOG = ROOT / 'reports/chexpert_plus_pairrank_v8_completion.log'
PAIR_PID = 309007
BASELINE = ROOT / 'checkpoints/chexpert_plus_findings_320_v7/best_metrics.json'
TARGET = 0.93
RUNS = [
    ('efficientnet_v2_s', 'checkpoints/chexpert_plus_efficientnet_v2_s_v9', 12),
    ('convnext_tiny', 'checkpoints/chexpert_plus_convnext_tiny_v10', 12),
]


def status(phase, **details):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    REPORT.write_text(json.dumps(payload, indent=2))
    print('STATUS', json.dumps(payload), flush=True)


def run(script, *arguments, log_name):
    log_path = ROOT / 'reports' / log_name
    with log_path.open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *arguments],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def wait_for_pairrank():
    start = time.monotonic()
    while time.monotonic() - start < 12 * 3600:
        if 'CHEXPERT_PAIRRANK_V8_COMPLETE' in PAIR_LOG.read_text():
            return 'completed'
        try:
            os.kill(PAIR_PID, 0)
        except ProcessLookupError:
            return 'watcher_ended_without_completion_marker'
        time.sleep(30)
    raise RuntimeError('Timed out waiting for the pairwise-ranking training job')


def main():
    status('waiting_for_pairrank_v8')
    prior = wait_for_pairrank()
    results = {}
    for architecture, output, batch_size in RUNS:
        status('training', architecture=architecture, prior_job=prior, completed=results)
        run('train_chexpert_alt_backbone_320.py', '--architecture', architecture,
            '--epochs', '4', '--batch-size', str(batch_size), '--workers', '8',
            log_name=f'chexpert_{architecture}_train.log')
        status('expert_validation', architecture=architecture, completed=results)
        run('evaluate_chexpert_alt_backbone_320.py', '--checkpoint-dir', output,
            log_name=f'chexpert_{architecture}_completion.log')
        checkpoint = ROOT / output / 'best_model.pth'
        development = json.loads((ROOT / output / 'best_metrics.json').read_text())
        expert = json.loads((ROOT / output / 'expert_valid_comparison.json').read_text())
        results[architecture] = {
            'checkpoint_dir': output,
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'findings_development_macro_auroc': development['five_finding_macro_auroc'],
            'expert_validation_macro_auroc_exploratory': expert['candidate']['macro_auroc'],
        }
    baseline = json.loads(BASELINE.read_text())['five_finding_macro_auroc']
    winner = max(results, key=lambda name: results[name]['findings_development_macro_auroc'])
    selected = results[winner]
    decision = {
        'selection_rule': 'Highest five-finding AUROC on frozen patient-disjoint Findings development set',
        'baseline_v7_development_auroc': baseline,
        'architecture_results': results,
        'selected_architecture': winner,
        'selected_checkpoint_sha256': selected['checkpoint_sha256'],
        'target_exploratory_macro_auroc': TARGET,
        'prior_test_result_known': True,
        'repeat_test_status': 'exploratory only; not a fresh SOTA claim',
    }
    decision_path = ROOT / 'reports/chexpert_alt_backbones_selection.json'
    decision_path.write_text(json.dumps(decision, indent=2))
    if selected['findings_development_macro_auroc'] > baseline:
        status('exploratory_repeat_test', architecture=winner, completed=results)
        run('evaluate_chexpert_test_500_exploratory_v7.py', '--checkpoint-dir',
            selected['checkpoint_dir'], '--image-size', '320',
            log_name=f'chexpert_{winner}_completion.log')
        test = json.loads((ROOT / selected['checkpoint_dir'] / 'test_500_exploratory.json').read_text())
        decision['exploratory_test_macro_auroc'] = test['macro_auroc']
        decision['fresh_test_required'] = test['macro_auroc'] >= TARGET
        decision_path.write_text(json.dumps(decision, indent=2))
        if decision['fresh_test_required']:
            status('fresh_test_required', architecture=winner, exploratory_auroc=test['macro_auroc'],
                   checkpoint_sha256=selected['checkpoint_sha256'])
            print('Fresh, untouched, independently labeled test cohort is required before a confirmatory claim.', flush=True)
            return
    else:
        decision['exploratory_test_skipped'] = 'No improvement on frozen development split'
        decision_path.write_text(json.dumps(decision, indent=2))
    status('complete_below_target_or_not_tested', decision=str(decision_path.relative_to(ROOT)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
