"""Select fixed DenseNet201 flip and blend variants on frozen development data."""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chexpert_test_500 import DATA as TEST_DATA, TestImages, locate_study
from scripts.train_chexpert_findings_320_v7 import Rows as Rows320
from scripts.train_chexpert_plus_uncertainty_v4 import ROOT as ROOT224, Rows as Rows224


ROOT320 = Path('data/chexpert_plus_cnn_impression_320_v1')
V6 = Path('checkpoints/chexpert_plus_densenet201_v6')
V12 = Path('checkpoints/chexpert_plus_impression_320_v12')
OUT = Path('reports/chexpert_densenet201_v12_tta_blends.json')
TEST_OUT = Path('reports/chexpert_densenet201_v12_tta_blends_test_exploratory.json')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def verify():
    for directory in (V6, V12):
        if json.loads((directory / 'progress.json').read_text())['phase'] != 'complete':
            raise RuntimeError(f'{directory} is incomplete')
    manifest = json.loads((ROOT320 / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT320 / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'High-resolution {name} split changed')
    split = json.loads((ROOT224 / 'development_split.json').read_text())
    if hashlib.sha256((ROOT224 / 'development.jsonl').read_bytes()).hexdigest() != split['sha256']['development']:
        raise RuntimeError('Original development split changed')
    expert_manifest = json.loads((ROOT224 / 'expert_valid_manifest.json').read_text())
    if hashlib.sha256((ROOT224 / 'expert_valid.jsonl').read_bytes()).hexdigest() != expert_manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed')
    names = json.loads((ROOT224 / 'manifest.json').read_text())['labels']
    return [names.index(name) for name in FINDINGS]


def predict(directory, loader, indices, flip=False):
    model = models.densenet201(weights=None, num_classes=14).cuda().eval()
    model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True))
    scores = []
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.cuda(non_blocking=True)
            score = model(images).sigmoid()
            if flip:
                score = (score + model(images.flip(-1)).sigmoid()) / 2
            scores.append(score[:, indices].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(scores)


def combine(v6, v12, tta):
    return {
        'v6': v6,
        'v12': v12,
        'v12_flip_tta': tta,
        'equal_v6_v12': (v6 + v12) / 2,
        'equal_v6_v12_tta': (v6 + tta) / 2,
    }


def metrics(y, scores):
    per = {}
    for j, name in enumerate(FINDINGS):
        keep = np.isin(y[:, j], [0, 1])
        truth = y[keep, j].astype(int)
        values = scores[keep, j]
        per[name] = {
            'positive': int(truth.sum()),
            'negative': int(len(truth) - truth.sum()),
            'auroc': float(roc_auc_score(truth, values)),
            'ap': float(average_precision_score(truth, values)),
        }
    return {
        'macro_auroc': float(np.mean([entry['auroc'] for entry in per.values()])),
        'macro_ap': float(np.mean([entry['ap'] for entry in per.values()])),
        'per_label': per,
    }


def cohort(name, indices):
    source = Rows224(ROOT224 / f'{name}.jsonl')
    high = Rows320(ROOT320 / f'{name}.jsonl')
    if [row['source_path'] for row in source.rows] != [row['source_path'] for row in high.rows]:
        raise RuntimeError(f'{name} image order mismatch')
    loader224 = DataLoader(source, batch_size=32, num_workers=4, pin_memory=True)
    loader320 = DataLoader(high, batch_size=16, num_workers=4, pin_memory=True)
    v6 = predict(V6, loader224, indices)
    print('PREDICTED', name, 'v6', len(v6), flush=True)
    v12 = predict(V12, loader320, indices, flip=True)
    # A separate single-view pass keeps the fixed TTA and single model comparable.
    single = predict(V12, loader320, indices)
    print('PREDICTED', name, 'v12+flip', len(v12), flush=True)
    return combine(v6, single, v12), source.rows


def test_scores(directory, size, indices, paths, flip):
    dataset = TestImages(paths)
    dataset.transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
    model = models.densenet201(weights=None, num_classes=14).cuda().eval()
    model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True))
    scores = []
    with torch.inference_mode():
        for images in loader:
            images = images.cuda(non_blocking=True)
            score = model(images).sigmoid()
            if flip:
                score = (score + model(images.flip(-1)).sigmoid()) / 2
            scores.append(score[:, indices].cpu().numpy())
    del model
    return np.concatenate(scores)


def exploratory_test(selected, indices):
    manifest = json.loads((TEST_DATA / 'manifest.json').read_text())
    raw = (TEST_DATA / 'groundtruth.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise RuntimeError('Test labels changed')
    studies = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(studies) != 500:
        raise RuntimeError('Expected 500 test studies')
    paths = []
    groups = defaultdict(list)
    for study_index, row in enumerate(studies):
        for path in locate_study(TEST_DATA / 'images', row['Study']):
            groups[study_index].append(len(paths))
            paths.append(path)
    if len(paths) != 518:
        raise RuntimeError('Expected 518 frontal test images')
    v6 = test_scores(V6, 224, indices, paths, False)
    single = test_scores(V12, 320, indices, paths, False)
    tta = test_scores(V12, 320, indices, paths, True)
    selected_scores = combine(v6, single, tta)[selected]
    scores = np.stack([selected_scores[groups[index]].mean(axis=0) for index in range(500)])
    y = np.array([[int(row[name]) for name in FINDINGS] for row in studies], dtype=int)
    result = metrics(y, scores)
    rng = np.random.default_rng(46)
    bootstrap = []
    for _ in range(2000):
        sample = rng.integers(0, 500, size=500)
        if any(np.unique(y[sample, j]).size != 2 for j in range(5)):
            continue
        bootstrap.append(float(np.mean([roc_auc_score(y[sample, j], scores[sample, j]) for j in range(5)])))
    result.update({
        'status': 'exploratory repeat evaluation on an already inspected CheXpert test cohort',
        'selected_by': 'frozen patient-disjoint development five-finding macro AUROC',
        'selected_variant': selected,
        'study_count': 500,
        'frontal_image_count': 518,
        'patient_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'warning': 'Cannot support a fresh leaderboard or SOTA claim; an independent test is required.',
    })
    TEST_OUT.write_text(json.dumps(result, indent=2))
    print('EXPLORATORY_TEST_AUROC', result['macro_auroc'], flush=True)


def main():
    torch.set_num_threads(4)
    indices = verify()
    development, dev_rows = cohort('development', indices)
    ydev = np.nan_to_num(np.array([row['labels'] for row in dev_rows], dtype=np.float32)[:, indices], nan=0)
    dev_results = {name: metrics(ydev, scores) for name, scores in development.items()}
    selected = max(dev_results, key=lambda name: dev_results[name]['macro_auroc'])
    print('DEVELOPMENT', {name: round(value['macro_auroc'], 6) for name, value in dev_results.items()}, flush=True)
    expert, expert_rows = cohort('valid', indices)
    expert_manifest = json.loads((ROOT224 / 'expert_valid_manifest.json').read_text())
    labels = [json.loads(line) for line in (ROOT224 / 'expert_valid.jsonl').read_text().splitlines()]
    if [row['source_path'] for row in labels] != [row['source_path'] for row in expert_rows]:
        raise RuntimeError('Expert image order mismatch')
    if expert_manifest['findings'] != FINDINGS:
        raise RuntimeError('Expert finding order mismatch')
    yexpert = np.array([[row['labels'][name] for name in FINDINGS] for row in labels], dtype=int)
    expert_results = {name: metrics(yexpert, scores) for name, scores in expert.items()}
    result = {
        'candidates': list(development),
        'selection_rule': 'highest five-finding macro AUROC on frozen 19688-image patient-disjoint Impression development split',
        'selected': selected,
        'development': dev_results,
        'expert_validation_exploratory': expert_results,
        'checkpoint_sha256': {name: hashlib.sha256((directory / 'best_model.pth').read_bytes()).hexdigest()
                              for name, directory in (('v6', V6), ('v12', V12))},
        'test_result_known_before_experiment': True,
        'test_gate': 'Selected variant must beat both v6 and v12 on development and expert validation',
    }
    OUT.write_text(json.dumps(result, indent=2))
    print('EXPERT', {name: round(value['macro_auroc'], 6) for name, value in expert_results.items()}, flush=True)
    print('SELECTED', selected, flush=True)
    if selected not in ('v6', 'v12') and all(
        dev_results[selected]['macro_auroc'] > dev_results[base]['macro_auroc']
        and expert_results[selected]['macro_auroc'] > expert_results[base]['macro_auroc']
        for base in ('v6', 'v12')
    ):
        exploratory_test(selected, indices)
    else:
        print('SKIPPED_REPEAT_TEST: no joint gain over both base models', flush=True)


if __name__ == '__main__':
    main()
