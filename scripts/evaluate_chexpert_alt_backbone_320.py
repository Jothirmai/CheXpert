"""Compare a 320-pixel alternative CNN with v7 on expert validation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bootstrap_chexpert_expert_valid import macro_auc
from scripts.evaluate_chexpert_expert_valid import cnn_scores, metrics
from scripts.train_chexpert_findings_320_v7 import ROOT, Rows
from scripts.train_chexpert_plus_uncertainty_v4 import ROOT as IMPRESSION_ROOT


BASELINE = Path('checkpoints/chexpert_plus_findings_320_v7')


def make_model(architecture):
    if architecture == 'efficientnet_v2_s':
        return models.efficientnet_v2_s(weights=None, num_classes=14)
    if architecture == 'convnext_tiny':
        return models.convnext_tiny(weights=None, num_classes=14)
    raise ValueError(architecture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--baseline-checkpoint-dir', type=Path, default=BASELINE)
    args = parser.parse_args()
    directory = args.checkpoint_dir
    data_root = args.data_root
    baseline_directory = args.baseline_checkpoint_dir
    protocol = json.loads((directory / 'protocol.json').read_text())
    if json.loads((directory / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Candidate training is incomplete')
    for name, digest in protocol['data_hashes'].items():
        if hashlib.sha256((data_root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen 320-pixel data changed: {name}')
    expert_manifest = json.loads((IMPRESSION_ROOT / 'expert_valid_manifest.json').read_text())
    label_file = IMPRESSION_ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(label_file.read_bytes()).hexdigest() != expert_manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed')
    expert = [json.loads(line) for line in label_file.read_text().splitlines()]
    rows = Rows(data_root / 'valid.jsonl')
    if [row['source_path'] for row in rows.rows] != [row['source_path'] for row in expert]:
        raise RuntimeError('Expert labels and 320-pixel images are out of order')
    names = json.loads((IMPRESSION_ROOT / 'manifest.json').read_text())['labels']
    findings = expert_manifest['findings']
    indices = {name: names.index(name) for name in findings}
    y = np.array([[row['labels'][name] for name in findings] for row in expert], dtype=int)
    patients = np.array([row['patient_id'] for row in expert])
    loader = DataLoader(rows, batch_size=16, num_workers=4, pin_memory=True)
    torch.set_num_threads(4)
    baseline = models.densenet201(weights=None, num_classes=14).cuda()
    baseline.load_state_dict(torch.load(baseline_directory / 'best_model.pth', map_location='cpu', weights_only=True))
    baseline_scores = cnn_scores(baseline, loader)
    del baseline
    model = make_model(protocol['architecture']).cuda()
    model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True))
    candidate_scores = cnn_scores(model, loader)
    baseline_metric = metrics(baseline_scores, expert, indices)
    candidate_metric = metrics(candidate_scores, expert, indices)
    baseline_five = baseline_scores[:, [indices[name] for name in findings]]
    candidate_five = candidate_scores[:, [indices[name] for name in findings]]
    unique = np.unique(patients)
    by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(46)
    differences = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([by_patient[patient] for patient in sampled])
        old = macro_auc(y, baseline_five, sample)
        new = macro_auc(y, candidate_five, sample)
        if old is not None and new is not None:
            differences.append(new - old)
    result = {
        'cohort': '202 expert-labeled CheXpert validation frontal images, 200 patients',
        'findings': findings,
        'baseline_checkpoint_dir': str(baseline_directory),
        'baseline': baseline_metric,
        'candidate': candidate_metric,
        'candidate_minus_baseline': candidate_metric['macro_auroc'] - baseline_metric['macro_auroc'],
        'paired_patient_bootstrap_95ci': np.quantile(differences, [0.025, 0.975]).tolist(),
        'bootstrap_replicates': len(differences),
        'limitation': 'Expert validation has been inspected for prior experiments; this comparison is exploratory.',
    }
    (directory / 'expert_valid_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'baseline': baseline_metric['macro_auroc'],
                      'candidate': candidate_metric['macro_auroc'],
                      'difference_95ci': result['paired_patient_bootstrap_95ci']}, indent=2))


if __name__ == '__main__':
    main()
