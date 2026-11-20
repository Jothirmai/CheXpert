"""Compare fixed frontal/lateral study fusion on disjoint development studies."""

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


FRONTAL = Path('data/chexpert_plus_cnn_impression_320_v1')
LATERAL = Path('data/chexpert_plus_cnn_lateral_320_v1')
V6 = Path('checkpoints/chexpert_plus_densenet201_v6')
V12 = Path('checkpoints/chexpert_plus_impression_320_v12')
V13 = Path('checkpoints/chexpert_plus_lateral_320_v13')
OUT = Path('reports/chexpert_multiview_v14.json')
TEST_OUT = Path('reports/chexpert_multiview_v14_test_exploratory.json')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
RULES = ('frontal_only', 'lateral_10_percent', 'lateral_25_percent',
         'lateral_50_percent', 'frontal_lateral_max')


def check_files():
    for checkpoint in (V6, V12, V13):
        if json.loads((checkpoint / 'progress.json').read_text()).get('phase') != 'complete':
            raise RuntimeError(f'Checkpoint incomplete: {checkpoint}')
    for root in (FRONTAL, LATERAL):
        manifest = json.loads((root / 'manifest.json').read_text())
        for split, digest in manifest['sha256'].items():
            if hashlib.sha256((root / f'{split}.jsonl').read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen split changed: {root}/{split}')
    source = json.loads((ROOT224 / 'manifest.json').read_text())['labels']
    return [source.index(name) for name in FINDINGS]


def predict(checkpoint, loader, indices, flip=False):
    model = models.densenet201(weights=None, num_classes=14).cuda().eval()
    model.load_state_dict(torch.load(checkpoint / 'best_model.pth', map_location='cpu', weights_only=True))
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.cuda(non_blocking=True)
            score = model(images).sigmoid()
            if flip:
                score = (score + model(images.flip(-1)).sigmoid()) / 2
            chunks.append(score[:, indices].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks)


def groups(rows):
    by = defaultdict(list)
    for index, row in enumerate(rows):
        by[row['source_path'].rsplit('/', 1)[0]].append(index)
    return by


def aggregate_studies(front_rows, front_scores, lateral_rows, lateral_scores, truth_rows):
    front_groups = groups(front_rows)
    lateral_groups = groups(lateral_rows)
    truth_groups = groups(truth_rows)
    studies = sorted(front_groups)
    candidates = {rule: [] for rule in RULES}
    truth = []
    with_lateral = 0
    excluded_discordant = 0
    for study in studies:
        truth_indices = truth_groups[study]
        labels = [truth_rows[index]['labels'] for index in truth_indices]
        if any(label != labels[0] for label in labels[1:]):
            excluded_discordant += 1
            continue
        frontal = front_scores[front_groups[study]].mean(axis=0)
        lateral = lateral_scores[lateral_groups[study]].mean(axis=0) if study in lateral_groups else None
        if lateral is not None:
            with_lateral += 1
        truth.append(labels[0])
        candidates['frontal_only'].append(frontal)
        for name, weight in (('lateral_10_percent', .1),
                             ('lateral_25_percent', .25),
                             ('lateral_50_percent', .5)):
            candidates[name].append(frontal if lateral is None else (1 - weight) * frontal + weight * lateral)
        candidates['frontal_lateral_max'].append(frontal if lateral is None else np.maximum(frontal, lateral))
    return {key: np.stack(values) for key, values in candidates.items()}, np.asarray(truth, dtype=float), {
        'studies': len(truth), 'with_lateral': with_lateral,
        'excluded_discordant_label_studies': excluded_discordant,
    }


def measure(truth, scores):
    per = {}
    for j, finding in enumerate(FINDINGS):
        keep = np.isin(truth[:, j], [0, 1])
        y = truth[keep, j].astype(int)
        s = scores[keep, j]
        per[finding] = {
            'positive': int(y.sum()), 'negative': int(len(y) - y.sum()),
            'auroc': float(roc_auc_score(y, s)),
            'ap': float(average_precision_score(y, s)),
        }
    return {
        'macro_auroc': float(np.mean([item['auroc'] for item in per.values()])),
        'macro_ap': float(np.mean([item['ap'] for item in per.values()])),
        'per_label': per,
    }


def evaluate_split(split, indices):
    original = Rows224(ROOT224 / f'{split}.jsonl')
    high = Rows320(FRONTAL / f'{split}.jsonl')
    lateral = Rows320(LATERAL / f'{split}.jsonl')
    if [row['source_path'] for row in original.rows] != [row['source_path'] for row in high.rows]:
        raise RuntimeError(f'Frontal image order mismatch: {split}')
    v6 = predict(V6, DataLoader(original, batch_size=32, num_workers=4, pin_memory=True), indices)
    print('PREDICTED', split, 'v6', len(v6), flush=True)
    v12 = predict(V12, DataLoader(high, batch_size=16, num_workers=4, pin_memory=True), indices, flip=True)
    print('PREDICTED', split, 'v12_tta', len(v12), flush=True)
    v13 = predict(V13, DataLoader(lateral, batch_size=16, num_workers=4, pin_memory=True), indices)
    print('PREDICTED', split, 'v13_lateral', len(v13), flush=True)
    front_score = (v6 + v12) / 2
    if split == 'development':
        truth_rows = []
        for row in original.rows:
            labels = np.nan_to_num(np.array(row['labels'], dtype=float)[indices], nan=0).tolist()
            truth_rows.append({'source_path': row['source_path'], 'labels': labels})
    else:
        manifest = json.loads((ROOT224 / 'expert_valid_manifest.json').read_text())
        path = ROOT224 / 'expert_valid.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['jsonl_sha256']:
            raise RuntimeError('Expert labels changed')
        expert = [json.loads(line) for line in path.read_text().splitlines()]
        if [row['source_path'] for row in expert] != [row['source_path'] for row in original.rows]:
            raise RuntimeError('Expert image order mismatch')
        truth_rows = [{'source_path': row['source_path'],
                       'labels': [row['labels'][name] for name in FINDINGS]} for row in expert]
    return aggregate_studies(original.rows, front_score, lateral.rows, v13, truth_rows)


def test_loader(paths, size):
    dataset = TestImages(paths)
    dataset.transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return DataLoader(dataset, batch_size=16 if size == 320 else 32,
                      num_workers=4, pin_memory=True)


def exploratory_test(selected, indices):
    manifest = json.loads((TEST_DATA / 'manifest.json').read_text())
    raw = (TEST_DATA / 'groundtruth.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise RuntimeError('Test labels changed')
    studies = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(studies) != 500:
        raise RuntimeError('Expected 500 test studies')
    frontal_paths = []
    lateral_paths = []
    front_group = defaultdict(list)
    lateral_group = defaultdict(list)
    for study_index, row in enumerate(studies):
        for path in locate_study(TEST_DATA / 'images', row['Study']):
            front_group[study_index].append(len(frontal_paths))
            frontal_paths.append(path)
        folder = TEST_DATA / 'images' / row['Study'].replace('CheXpert-v1.0/', 'CheXpert/')
        for path in sorted(folder.glob('*lateral*')):
            lateral_group[study_index].append(len(lateral_paths))
            lateral_paths.append(path)
    if len(frontal_paths) != 518 or len(lateral_paths) != 150:
        raise RuntimeError('Frozen test image cohort changed')
    v6 = predict(V6, test_loader(frontal_paths, 224), indices)
    v12 = predict(V12, test_loader(frontal_paths, 320), indices, flip=True)
    v13 = predict(V13, test_loader(lateral_paths, 320), indices)
    front = (v6 + v12) / 2
    scores = []
    for i in range(500):
        base = front[front_group[i]].mean(axis=0)
        lateral = v13[lateral_group[i]].mean(axis=0) if i in lateral_group else None
        if selected == 'frontal_only' or lateral is None:
            scores.append(base)
        elif selected == 'frontal_lateral_max':
            scores.append(np.maximum(base, lateral))
        else:
            weight = {'lateral_10_percent': .1, 'lateral_25_percent': .25,
                      'lateral_50_percent': .5}[selected]
            scores.append((1 - weight) * base + weight * lateral)
    scores = np.stack(scores)
    y = np.array([[int(row[name]) for name in FINDINGS] for row in studies])
    result = measure(y, scores)
    rng = np.random.default_rng(49)
    bootstrap = []
    for _ in range(2000):
        sample = rng.integers(0, 500, size=500)
        if any(np.unique(y[sample, j]).size != 2 for j in range(5)):
            continue
        bootstrap.append(float(np.mean([roc_auc_score(y[sample, j], scores[sample, j]) for j in range(5)])))
    result.update({
        'status': 'exploratory repeat test; prior test results were known',
        'selected_variant': selected,
        'study_count': 500,
        'studies_with_lateral': len(lateral_group),
        'patient_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'warning': 'Cannot support a fresh leaderboard or SOTA claim; a new independent test is required.',
    })
    TEST_OUT.write_text(json.dumps(result, indent=2))
    print('EXPLORATORY_TEST_AUROC', result['macro_auroc'], flush=True)


def main():
    torch.set_num_threads(4)
    indices = check_files()
    development, ydev, dev_counts = evaluate_split('development', indices)
    dev = {name: measure(ydev, values) for name, values in development.items()}
    selected = max(RULES, key=lambda name: dev[name]['macro_auroc'])
    print('DEVELOPMENT', {name: round(value['macro_auroc'], 6) for name, value in dev.items()}, flush=True)
    expert, yexpert, expert_counts = evaluate_split('valid', indices)
    valid = {name: measure(yexpert, values) for name, values in expert.items()}
    print('EXPERT', {name: round(value['macro_auroc'], 6) for name, value in valid.items()}, flush=True)
    result = {
        'selection_rule': 'Highest five-finding study-level mean AUROC on patient-disjoint development studies',
        'selected': selected,
        'candidate_rules': RULES,
        'development_cohort': dev_counts,
        'expert_validation_cohort': expert_counts,
        'development': dev,
        'expert_validation_exploratory': valid,
        'checkpoint_sha256': {name: hashlib.sha256((directory / 'best_model.pth').read_bytes()).hexdigest()
                              for name, directory in (('v6', V6), ('v12', V12), ('v13_lateral', V13))},
        'test_result_known_before_experiment': True,
        'test_gate': 'Selected multi-view rule must improve development and expert validation over frontal-only baseline',
    }
    OUT.write_text(json.dumps(result, indent=2))
    print('SELECTED', selected, flush=True)
    if selected != 'frontal_only' and (dev[selected]['macro_auroc'] > dev['frontal_only']['macro_auroc']
                                       and valid[selected]['macro_auroc'] > valid['frontal_only']['macro_auroc']):
        exploratory_test(selected, indices)
    else:
        print('SKIPPED_REPEAT_TEST: no joint development and expert-validation gain', flush=True)


if __name__ == '__main__':
    main()
