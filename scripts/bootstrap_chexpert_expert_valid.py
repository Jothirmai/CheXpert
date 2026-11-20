"""Paired patient bootstrap comparing a CNN candidate with released XRV all weights."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chexpert_expert_valid import OUT, ROOT, XRVImages, cnn_scores, xrv_scores
from scripts.train_chexpert_plus_impression import Rows


def macro_auc(y, scores, indices):
    values = []
    for j in range(y.shape[1]):
        truth = y[indices, j]
        if truth.min() == truth.max():
            return None
        values.append(roc_auc_score(truth, scores[indices, j]))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-out', default=str(OUT))
    args = parser.parse_args()
    candidate_out = Path(args.candidate_out)
    protocol = json.loads((candidate_out / 'protocol.json').read_text())
    if json.loads((candidate_out / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Candidate training is incomplete')
    training_root = Path('data/chexpert_plus_cnn_findings_subset_v1') if protocol['label_file'] == 'findings_fixed.json' else ROOT
    for split, digest in protocol['data_hashes'].items():
        assert hashlib.sha256((training_root / f'{split}.jsonl').read_bytes()).hexdigest() == digest
    expert_manifest = json.loads((ROOT / 'expert_valid_manifest.json').read_text())
    path = ROOT / 'expert_valid.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expert_manifest['jsonl_sha256']
    expert = [json.loads(line) for line in path.read_text().splitlines()]
    rows = Rows(ROOT / 'valid.jsonl')
    assert [row['source_path'] for row in expert] == [row['source_path'] for row in rows.rows]
    findings = expert_manifest['findings']
    y = np.array([[row['labels'][finding] for finding in findings] for row in expert], dtype=int)
    patients = np.array([row['patient_id'] for row in expert])
    unique = np.unique(patients)
    by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    torch.set_num_threads(4)
    is_densenet201 = candidate_out.name in ('chexpert_plus_densenet201_v6', 'chexpert_plus_pairrank_v8')
    model = (models.densenet201(weights=None, num_classes=14).cuda()
             if is_densenet201
             else models.densenet121(weights=None, num_classes=14).cuda())
    model.load_state_dict(torch.load(candidate_out / 'best_model.pth', map_location='cpu', weights_only=True))
    ours_full = cnn_scores(model, DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True))
    labels = json.loads((ROOT / 'manifest.json').read_text())['labels']
    ours = ours_full[:, [labels.index(finding) for finding in findings]]
    source = None
    if candidate_out.name == 'chexpert_plus_pairrank_v8':
        source_dir = Path('checkpoints/chexpert_plus_densenet201_v6')
        model.load_state_dict(torch.load(source_dir / 'best_model.pth', map_location='cpu', weights_only=True))
        source_full = cnn_scores(model, DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True))
        source = source_full[:, [labels.index(finding) for finding in findings]]
    prior = None
    if candidate_out != OUT:
        prior_model = (models.densenet121(weights=None, num_classes=14).cuda()
                       if is_densenet201 else model)
        prior_model.load_state_dict(torch.load(OUT / 'best_model.pth', map_location='cpu', weights_only=True))
        prior_full = cnn_scores(prior_model, DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True))
        prior = prior_full[:, [labels.index(finding) for finding in findings]]
        if prior_model is not model:
            del prior_model
    del model
    xmodel = xrv.models.DenseNet(weights='densenet121-res224-all').cuda()
    xrv_full = xrv_scores(xmodel, DataLoader(XRVImages(rows.rows), batch_size=32, num_workers=4, pin_memory=True))
    xrv_array = xrv_full[:, [xmodel.pathologies.index('Effusion' if finding == 'Pleural Effusion' else finding) for finding in findings]]
    full = np.arange(len(expert))
    ours_auc = macro_auc(y, ours, full)
    xrv_auc = macro_auc(y, xrv_array, full)
    fixed_blend = 0.5 * ours + 0.5 * xrv_array
    blend_auc = macro_auc(y, fixed_blend, full)
    rng = np.random.default_rng(42)
    differences = []
    prior_differences = []
    source_differences = []
    for _ in range(2000):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in sample])
        first = macro_auc(y, xrv_array, indices)
        second = macro_auc(y, ours, indices)
        if first is not None and second is not None:
            differences.append(second - first)
        if prior is not None:
            prior_auc = macro_auc(y, prior, indices)
            if prior_auc is not None and second is not None:
                prior_differences.append(second - prior_auc)
        if source is not None:
            source_auc = macro_auc(y, source, indices)
            if source_auc is not None and second is not None:
                source_differences.append(second - source_auc)
    result = {
        'cohort': 'CheXpert expert validation, 202 frontal images, 200 patients',
        'findings': findings,
        'candidate_macro_auroc': ours_auc,
        'xrv_all_macro_auroc': xrv_auc,
        'fixed_50_50_blend_macro_auroc_exploratory': blend_auc,
        'paired_difference': ours_auc - xrv_auc,
        'patient_bootstrap_95ci': np.quantile(differences, [0.025, 0.975]).tolist(),
        'replicates': len(differences),
        'bootstrap_seed': 42,
        'limitation': 'The expert validation subset is not the private CheXpert competition test set.',
    }
    if prior is not None:
        prior_auc = macro_auc(y, prior, full)
        result['v3_macro_auroc'] = prior_auc
        result['candidate_minus_v3'] = ours_auc - prior_auc
        result['candidate_minus_v3_95ci'] = np.quantile(prior_differences, [0.025, 0.975]).tolist()
    if source is not None:
        source_auc = macro_auc(y, source, full)
        result['v6_macro_auroc'] = source_auc
        result['candidate_minus_v6'] = ours_auc - source_auc
        result['candidate_minus_v6_95ci'] = np.quantile(source_differences, [0.025, 0.975]).tolist()
    (candidate_out / 'expert_valid_bootstrap.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
