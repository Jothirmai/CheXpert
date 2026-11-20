"""Select a fixed DenseNet201 probability blend on patient-disjoint development data."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chexpert_expert_valid import cnn_scores
from scripts.train_chexpert_findings_320_v7 import ROOT as ROOT320, Rows as Rows320
from scripts.train_chexpert_plus_impression import ROOT
from scripts.train_chexpert_plus_uncertainty_v4 import Rows as BaseRows


OUT = Path('reports/chexpert_densenet201_ensemble_v11.json')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
SOURCES = {
    'v6': ('checkpoints/chexpert_plus_densenet201_v6', 224),
    'v7': ('checkpoints/chexpert_plus_findings_320_v7', 320),
    'v8': ('checkpoints/chexpert_plus_pairrank_v8', 224),
}
BLENDS = {
    'v6': {'v6': 1},
    'v7': {'v7': 1},
    'v8': {'v8': 1},
    'equal_v6_v7': {'v6': .5, 'v7': .5},
    'equal_v6_v8': {'v6': .5, 'v8': .5},
    'equal_v6_v7_v8': {'v6': 1/3, 'v7': 1/3, 'v8': 1/3},
}


class Rows224(BaseRows):
    def __init__(self, path):
        super().__init__(path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])


def score_metrics(y, scores):
    per = {}
    for j, finding in enumerate(FINDINGS):
        keep = np.isin(y[:, j], [0, 1])
        truth = y[keep, j].astype(int)
        values = scores[keep, j]
        per[finding] = {
            'positive': int(truth.sum()),
            'negative': int(len(truth) - truth.sum()),
            'auroc': float(roc_auc_score(truth, values)),
            'ap': float(average_precision_score(truth, values)),
        }
    return {
        'macro_auroc': float(np.mean([item['auroc'] for item in per.values()])),
        'macro_ap': float(np.mean([item['ap'] for item in per.values()])),
        'per_label': per,
    }


def predict(cohort, source, indices):
    directory, size = SOURCES[source]
    directory = Path(directory)
    if json.loads((directory / 'progress.json').read_text())['phase'] != 'complete':
        raise RuntimeError(f'{source} checkpoint incomplete')
    protocol = json.loads((directory / 'protocol.json').read_text())
    rows = (Rows320 if size == 320 else Rows224)(ROOT320 / f'{cohort}.jsonl')
    loader = DataLoader(rows, batch_size=32 if size == 224 else 16,
                        num_workers=4, pin_memory=True)
    model = models.densenet201(weights=None, num_classes=14).cuda()
    checkpoint = directory / 'best_model.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    scores = cnn_scores(model, loader)[:, indices]
    print('PREDICTED', cohort, source, len(scores), flush=True)
    del model
    torch.cuda.empty_cache()
    return scores, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), protocol


def main():
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    indices = [manifest['labels'].index(name) for name in FINDINGS]
    development = ROOT320 / 'development.jsonl'
    split = json.loads((ROOT320 / 'manifest.json').read_text())
    if hashlib.sha256(development.read_bytes()).hexdigest() != split['sha256']['development']:
        raise RuntimeError('Development data changed')
    expert_manifest = json.loads((ROOT / 'expert_valid_manifest.json').read_text())
    expert_path = ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(expert_path.read_bytes()).hexdigest() != expert_manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed')
    dev_rows = Rows320(development).rows
    dev_y = np.nan_to_num(np.array([row['labels'] for row in dev_rows], dtype=np.float32)[:, indices], nan=0)
    expert_rows = [json.loads(line) for line in expert_path.read_text().splitlines()]
    valid_rows = Rows320(ROOT320 / 'valid.jsonl').rows
    if [row['source_path'] for row in expert_rows] != [row['source_path'] for row in valid_rows]:
        raise RuntimeError('Expert image order changed')
    expert_y = np.array([[row['labels'][name] for name in FINDINGS] for row in expert_rows], dtype=np.float32)
    prediction = {}
    hashes = {}
    for cohort in ('development', 'valid'):
        prediction[cohort] = {}
        for source in SOURCES:
            prediction[cohort][source], hashes[source], _ = predict(cohort, source, indices)
    measured = {}
    for cohort, truth in (('development', dev_y), ('expert_validation', expert_y)):
        pred = prediction['development' if cohort == 'development' else 'valid']
        measured[cohort] = {
            name: score_metrics(truth, sum(weight * pred[source] for source, weight in weights.items()))
            for name, weights in BLENDS.items()
        }
    selected = max(BLENDS, key=lambda name: measured['development'][name]['macro_auroc'])
    result = {
        'protocol': 'Predeclared fixed probability blends; select by five-finding macro AUROC on frozen patient-disjoint Findings development split of 4934 images',
        'findings': FINDINGS,
        'blends': BLENDS,
        'source_checkpoint_sha256': hashes,
        'selection': selected,
        'test_result_known_before_experiment': True,
        'expert_validation_already_inspected': True,
        'development': measured['development'],
        'expert_validation_exploratory': measured['expert_validation'],
        'repeat_chexpert_test_policy': 'Only consider an exploratory repeat test if selected blend improves both development and expert validation over v6; a fresh external test is needed for a confirmatory claim.',
    }
    OUT.write_text(json.dumps(result, indent=2))
    print('SELECTED', selected, flush=True)
    for cohort in ('development', 'expert_validation_exploratory'):
        print(cohort, {name: round(item['macro_auroc'], 6) for name, item in result[cohort].items()}, flush=True)


if __name__ == '__main__':
    main()
