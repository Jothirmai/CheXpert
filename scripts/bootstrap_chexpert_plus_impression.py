"""Paired patient bootstrap for the five common CheXpert findings on official valid."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_plus_impression import ROOT, Rows

OUT = Path('checkpoints/chexpert_plus_impression_v3')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def predict(model, loader):
    model.eval()
    batches = []
    with torch.inference_mode():
        for x, _, _ in loader:
            batches.append(model(x.cuda(non_blocking=True)).sigmoid().cpu().numpy())
    return np.concatenate(batches)


def macro_auc(y, mask, score, indices, selected):
    values = []
    for j in selected:
        use = indices[mask[indices, j]]
        yy = y[use, j]
        if len(yy) == 0 or yy.min() == yy.max():
            return None
        values.append(roc_auc_score(yy, score[use, j]))
    return float(np.mean(values))


def main():
    progress = json.loads((OUT / 'progress.json').read_text())
    if progress.get('phase') != 'complete':
        raise RuntimeError('CNN training has not completed')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for split, digest in protocol['data_hashes'].items():
        assert hashlib.sha256((ROOT / f'{split}.jsonl').read_bytes()).hexdigest() == digest
    data = Rows(ROOT / 'valid.jsonl')
    labels = json.loads((ROOT / 'manifest.json').read_text())['labels']
    selected = [labels.index(label) for label in FINDINGS]
    raw = np.array([[np.nan if v is None else v for v in row['labels']] for row in data.rows], dtype=float)
    y = np.nan_to_num(raw, nan=0).clip(0, 1)
    mask = raw != -1
    patients = np.array([row['patient_id'] for row in data.rows])
    unique = np.unique(patients)
    by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    loader = DataLoader(data, batch_size=32, num_workers=4, pin_memory=True)
    torch.set_num_threads(4)
    model = models.densenet121(weights=None, num_classes=14).cuda()
    original = torch.load('/home/doli/HCT_LSR/hct14_ensemble/models/densenet121_hct_lsr_14_stage1_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(original['model_state_dict'])
    before = predict(model, loader)
    model.load_state_dict(torch.load(OUT / 'best_model.pth', map_location='cpu', weights_only=True))
    after = predict(model, loader)
    full = np.arange(len(data))
    original_auc = macro_auc(y, mask, before, full, selected)
    candidate_auc = macro_auc(y, mask, after, full, selected)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        sample = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in sample])
        first = macro_auc(y, mask, before, indices, selected)
        second = macro_auc(y, mask, after, indices, selected)
        if first is not None and second is not None:
            differences.append(second - first)
    result = {
        'cohort': 'CheXpert Plus official valid; 202 frontal images, 200 patients',
        'labels': FINDINGS,
        'label_policy': 'blank negative; uncertain ignored',
        'original_macro_auroc': original_auc,
        'candidate_macro_auroc': candidate_auc,
        'difference': candidate_auc - original_auc,
        'paired_patient_bootstrap_95ci': np.quantile(differences, [0.025, 0.975]).tolist(),
        'valid_bootstrap_replicates': len(differences),
        'bootstrap_seed': 42,
        'limitations': 'Weak report labels and small official validation; not the expert-labeled CheXpert competition test.',
    }
    (OUT / 'bootstrap_five_findings.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
