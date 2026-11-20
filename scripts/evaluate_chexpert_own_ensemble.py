"""Evaluate fixed blends of our CNN checkpoints on common development and expert sets."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chexpert_expert_valid import cnn_scores
from scripts.train_chexpert_plus_impression import ROOT, Rows


CHECKPOINTS = {
    'v3_impression': Path('checkpoints/chexpert_plus_impression_v3'),
    'v4_uncertainty': Path('checkpoints/chexpert_plus_uncertainty_v4'),
    'v5_findings': Path('checkpoints/chexpert_plus_findings_v5'),
}
FINDINGS_ROOT = Path('data/chexpert_plus_cnn_findings_subset_v1')
OUT = Path('reports/sota_comparison/CHEXPERT_OWN_ENSEMBLE.json')


def score_metrics(y, scores):
    per = {}
    for j, finding in enumerate(FINDINGS):
        mask = np.isin(y[:, j], [0, 1])
        truth = y[mask, j].astype(int)
        if truth.min() == truth.max():
            raise RuntimeError(f'No positive/negative pairs for {finding}')
        values = scores[mask, j]
        per[finding] = {
            'positive': int(truth.sum()),
            'negative': int(len(truth) - truth.sum()),
            'auroc': float(roc_auc_score(truth, values)),
            'ap': float(average_precision_score(truth, values)),
        }
    return {
        'macro_auroc': float(np.mean([v['auroc'] for v in per.values()])),
        'macro_ap': float(np.mean([v['ap'] for v in per.values()])),
        'per_label': per,
    }


def predict_all(rows):
    loader = DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True)
    model = models.densenet121(weights=None, num_classes=14).cuda()
    predictions = {}
    for name, directory in CHECKPOINTS.items():
        protocol = json.loads((directory / 'protocol.json').read_text())
        root = FINDINGS_ROOT if protocol['label_file'] == 'findings_fixed.json' else ROOT
        for split, digest in protocol['data_hashes'].items():
            if hashlib.sha256((root / f'{split}.jsonl').read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Training manifest changed for {name}:{split}')
        if json.loads((directory / 'progress.json').read_text()).get('phase') != 'complete':
            raise RuntimeError(f'{name} training is incomplete')
        checkpoint = directory / 'best_model.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        predictions[name] = cnn_scores(model, loader)[:, INDICES]
        print('Predicted', name, len(rows), flush=True)
    del model
    predictions['equal_v4_v5'] = (predictions['v4_uncertainty'] + predictions['v5_findings']) / 2
    predictions['equal_v3_v4_v5'] = sum(predictions[name] for name in CHECKPOINTS) / 3
    return predictions


if __name__ == '__main__':
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    expert_manifest = json.loads((ROOT / 'expert_valid_manifest.json').read_text())
    FINDINGS = expert_manifest['findings']
    INDICES = [manifest['labels'].index(name) for name in FINDINGS]
    expert_path = ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(expert_path.read_bytes()).hexdigest() != expert_manifest['jsonl_sha256']:
        raise RuntimeError('Expert label file changed')
    dev_rows = Rows(ROOT / 'development.jsonl')
    expert_rows = Rows(ROOT / 'valid.jsonl')
    expert_labels = [json.loads(line) for line in expert_path.read_text().splitlines()]
    if [row['source_path'] for row in expert_rows.rows] != [row['source_path'] for row in expert_labels]:
        raise RuntimeError('Expert images and labels do not align')
    dev_y = np.array([row['labels'] for row in dev_rows.rows], dtype=np.float32)[:, INDICES]
    # Match the frozen development policy: blanks are negatives; explicit -1 is unknown.
    dev_y = np.nan_to_num(dev_y, nan=0.0)
    expert_y = np.array([[row['labels'][name] for name in FINDINGS] for row in expert_labels], dtype=np.float32)
    result = {
        'cohorts': {
            'development': '19,688 patient-disjoint CheXpert Plus train images; impression-derived labels',
            'expert_validation': '202 frontal images, 200 patients, CheXpert expert validation labels',
        },
        'findings': FINDINGS,
        'blend_policy': 'unweighted mean of probabilities fixed in advance; no expert-label weight fitting',
        'development': {name: score_metrics(dev_y, score) for name, score in predict_all(dev_rows).items()},
        'expert_validation': {name: score_metrics(expert_y, score) for name, score in predict_all(expert_rows).items()},
        'limitations': 'Expert validation was inspected during earlier model development. These ensemble results are exploratory and cannot establish leaderboard rank.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    for cohort in ('development', 'expert_validation'):
        print(cohort, {name: round(value['macro_auroc'], 6) for name, value in result[cohort].items()}, flush=True)
