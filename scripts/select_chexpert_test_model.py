"""Freeze one CNN for the CheXpert expert test using development scores only."""

import hashlib
import json
from pathlib import Path


ROOT = Path('data/chexpert_test_500')
CANDIDATES = {
    'densenet121_findings_v5': Path('checkpoints/chexpert_plus_findings_v5'),
    'densenet201_v6': Path('checkpoints/chexpert_plus_densenet201_v6'),
}


def main():
    target = ROOT / 'model_selection.json'
    if target.exists():
        raise RuntimeError('Test model selection is already frozen')
    scores = {}
    for name, directory in CANDIDATES.items():
        data = json.loads((directory / 'impression_development_five.json').read_text())
        checkpoint = directory / 'best_model.pth'
        if data['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError(f'Development score does not match {name} checkpoint')
        if json.loads((directory / 'progress.json').read_text()).get('phase') != 'complete':
            raise RuntimeError(f'{name} training is incomplete')
        scores[name] = {'macro_auroc': data['macro_auroc'],
                        'macro_ap': data['macro_ap'],
                        'checkpoint': str(checkpoint),
                        'checkpoint_sha256': data['checkpoint_sha256']}
    selected = max(scores, key=lambda name: (scores[name]['macro_auroc'], scores[name]['macro_ap']))
    result = {
        'policy': 'highest five-finding AUROC on the common patient-disjoint impression development set; AP breaks ties',
        'test_labels_consulted': False,
        'selected': selected,
        'candidates': scores,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
