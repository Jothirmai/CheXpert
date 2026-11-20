"""Run full-fit 320-pixel DenseNet201 training and gated exploratory evaluation."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/chexpert_plus_cnn_impression_320_v1'
OUT = ROOT / 'checkpoints/chexpert_plus_impression_320_v12'
SOURCE = ROOT / 'checkpoints/chexpert_plus_densenet201_v6'
STATUS = ROOT / 'reports/chexpert_plus_impression_320_v12_status.json'


def update(phase, **details):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(payload, indent=2))
    print('STATUS', json.dumps(payload), flush=True)


def run(script, *arguments):
    subprocess.run([sys.executable, '-u', f'scripts/{script}', *arguments], cwd=ROOT, check=True)


def main():
    update('preparing_high_resolution_images')
    if not (DATA / 'manifest.json').exists():
        run('prepare_chexpert_impression_320.py')
    update('training', fit_images=json.loads((DATA / 'manifest.json').read_text())['counts']['fit']['images'])
    if not (OUT / 'protocol.json').exists():
        run('train_chexpert_findings_320_v7.py', '--data-root', str(DATA.relative_to(ROOT)),
            '--output', str(OUT.relative_to(ROOT)), '--epochs', '2', '--batch-size', '12', '--workers', '8')
    if json.loads((OUT / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Training did not complete')
    update('expert_validation')
    run('evaluate_chexpert_findings_320_v7.py', '--data-root', str(DATA.relative_to(ROOT)),
        '--output', str(OUT.relative_to(ROOT)))
    development = json.loads((OUT / 'best_metrics.json').read_text())['five_finding_macro_auroc']
    source_development = json.loads((SOURCE / 'best_metrics.json').read_text())['five_finding_macro_auroc']
    expert = json.loads((OUT / 'expert_valid_comparison.json').read_text())
    candidate_expert = expert['candidate_320']['macro_auroc']
    source_expert = expert['source_224']['macro_auroc']
    decision = {
        'candidate_development_macro_auroc': development,
        'source_development_macro_auroc': source_development,
        'candidate_expert_validation_macro_auroc_exploratory': candidate_expert,
        'source_expert_validation_macro_auroc': source_expert,
        'repeat_test_gate': 'Candidate must improve development and expert validation over v6',
        'prior_test_result_known': True,
    }
    (OUT / 'selection.json').write_text(json.dumps(decision, indent=2))
    if development > source_development and candidate_expert > source_expert:
        update('exploratory_repeat_test', **decision)
        run('evaluate_chexpert_test_500_exploratory_v7.py', '--checkpoint-dir',
            str(OUT.relative_to(ROOT)), '--image-size', '320')
        result = json.loads((OUT / 'test_500_exploratory.json').read_text())
        update('fresh_test_required' if result['macro_auroc'] >= .93 else 'complete_below_target',
               exploratory_test_macro_auroc=result['macro_auroc'], **decision)
    else:
        update('complete_no_joint_gain', **decision)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update('failed', error=repr(error))
        raise
