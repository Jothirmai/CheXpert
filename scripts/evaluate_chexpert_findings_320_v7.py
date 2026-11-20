"""Compare the 320-pixel CNN with its 224-pixel source on expert validation."""

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
from scripts.train_chexpert_plus_uncertainty_v4 import ROOT as IMPRESSION_ROOT, Rows as BaseRows
from scripts.train_chexpert_findings_320_v7 import ROOT, Rows


OUT = Path('checkpoints/chexpert_plus_findings_320_v7')
SOURCE = Path('checkpoints/chexpert_plus_densenet201_v6')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    data_root = args.data_root
    out = args.output
    protocol = json.loads((out / 'protocol.json').read_text())
    if json.loads((out / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('320-pixel CNN training is incomplete')
    for name, digest in protocol['data_hashes'].items():
        if hashlib.sha256((data_root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen 320-pixel data changed: {name}')
    original_manifest = json.loads((IMPRESSION_ROOT / 'expert_valid_manifest.json').read_text())
    original_path = IMPRESSION_ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != original_manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed')
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    highres = Rows(data_root / 'valid.jsonl')
    lowres = BaseRows(IMPRESSION_ROOT / 'valid.jsonl')
    if [r['source_path'] for r in highres.rows] != [r['source_path'] for r in original]:
        raise RuntimeError('320-pixel validation images and expert labels are out of order')
    if [r['source_path'] for r in lowres.rows] != [r['source_path'] for r in original]:
        raise RuntimeError('224-pixel validation images and expert labels are out of order')
    names = json.loads((IMPRESSION_ROOT / 'manifest.json').read_text())['labels']
    findings = original_manifest['findings']
    indices = {name: names.index(name) for name in findings}
    targets = np.array([[row['labels'][name] for name in findings] for row in original], dtype=int)
    patients = np.array([row['patient_id'] for row in original])
    torch.set_num_threads(4)
    model = models.densenet201(weights=None, num_classes=14).cuda()
    model.load_state_dict(torch.load(SOURCE / 'best_model.pth', map_location='cpu', weights_only=True))
    source_scores = cnn_scores(model, DataLoader(lowres, batch_size=32, num_workers=4, pin_memory=True))
    model.load_state_dict(torch.load(out / 'best_model.pth', map_location='cpu', weights_only=True))
    candidate_scores = cnn_scores(model, DataLoader(highres, batch_size=16, num_workers=4, pin_memory=True))
    source = metrics(source_scores, original, indices)
    candidate = metrics(candidate_scores, original, indices)
    source_five = source_scores[:, [indices[name] for name in findings]]
    candidate_five = candidate_scores[:, [indices[name] for name in findings]]
    unique = np.unique(patients)
    by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(44)
    differences = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_patient[patient] for patient in sampled])
        old = macro_auc(targets, source_five, rows)
        new = macro_auc(targets, candidate_five, rows)
        if old is not None and new is not None:
            differences.append(new - old)
    result = {
        'cohort': '202 frontal expert-labeled CheXpert validation images, 200 patients',
        'findings': findings,
        'source_224': source,
        'candidate_320': candidate,
        'candidate_minus_source': candidate['macro_auroc'] - source['macro_auroc'],
        'paired_patient_bootstrap_95ci': np.quantile(differences, [0.025, 0.975]).tolist(),
        'bootstrap_replicates': len(differences),
        'limitation': 'The official CheXpert test result was known before this 320-pixel experiment; any later result on that test is exploratory.',
    }
    (out / 'expert_valid_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'source_224': source['macro_auroc'],
                      'candidate_320': candidate['macro_auroc'],
                      'difference_95ci': result['paired_patient_bootstrap_95ci']}, indent=2))


if __name__ == '__main__':
    main()
