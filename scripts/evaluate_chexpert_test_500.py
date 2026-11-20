"""Evaluate frozen CNN checkpoints by study on the public CheXpert expert test set.

The test images must be obtained under Stanford AIMI's access terms. This script
never trains or selects a checkpoint using test labels.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATA = Path('data/chexpert_test_500')
LABELS = json.loads(Path('data/chexpert_plus_cnn_impression_v1/manifest.json').read_text())['labels']


class TestImages(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def locate_study(images_root, relative):
    suffix = relative.removeprefix('CheXpert-v1.0/')
    candidates = [images_root / suffix, images_root / 'CheXpert' / suffix,
                  images_root / 'CheXpert-v1.0' / suffix]
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) != 1:
        raise FileNotFoundError(f'Expected one image directory for {relative}; found {len(existing)}')
    frontal = sorted(path for path in existing[0].iterdir()
                     if path.is_file() and 'frontal' in path.name.lower()
                     and path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not frontal:
        raise FileNotFoundError(f'No frontal image for {relative}')
    return frontal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-root', type=Path, default=DATA / 'images')
    parser.add_argument('--output', type=Path, default=DATA / 'evaluation.json')
    args = parser.parse_args()
    selection_path = DATA / 'model_selection.json'
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if selection['test_labels_consulted'] is not False:
        raise RuntimeError('Model selection must precede test-label evaluation')
    selected = selection['selected']
    checkpoint = Path(selection['candidates'][selected]['checkpoint'])
    checkpoint_hash = selection['candidates'][selected]['checkpoint_sha256']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError('Selected checkpoint changed after development selection')
    directory = checkpoint.parent
    if json.loads((directory / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Selected model training is incomplete')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    images_manifest = json.loads((DATA / 'image_manifest.json').read_text())
    if images_manifest['images'] != 668 or images_manifest['groundtruth_sha256'] != manifest['sha256']:
        raise RuntimeError('Test images and expert labels do not match the frozen cohort')
    raw = (DATA / 'groundtruth.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise RuntimeError('Frozen test labels changed')
    labels = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(labels) != 500 or len({row['Study'] for row in labels}) != 500:
        raise RuntimeError('Expected 500 distinct test studies')
    paths = []
    study_indices = defaultdict(list)
    for study_index, row in enumerate(labels):
        for path in locate_study(args.images_root, row['Study']):
            study_indices[study_index].append(len(paths))
            paths.append(path)
    if len(study_indices) != 500:
        raise RuntimeError('Test image cohort is incomplete')
    findings = manifest['findings']
    targets = np.array([[int(row[name]) for name in findings] for row in labels], dtype=int)
    loader = DataLoader(TestImages(paths), batch_size=32, num_workers=4, pin_memory=True)
    torch.set_num_threads(4)
    model = (models.densenet201(weights=None, num_classes=14)
             if selected == 'densenet201_v6' else models.densenet121(weights=None, num_classes=14))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.cuda().eval()
    chunks = []
    with torch.inference_mode():
        for x in loader:
            chunks.append(model(x.cuda(non_blocking=True)).sigmoid().cpu().numpy())
    image_scores = np.concatenate(chunks)
    study_scores = np.stack([image_scores[study_indices[i]].mean(axis=0) for i in range(500)])
    per = {}
    for j, finding in enumerate(findings):
        values = study_scores[:, LABELS.index(finding)]
        truth = targets[:, j]
        per[finding] = {
            'auroc': float(roc_auc_score(truth, values)),
            'ap': float(average_precision_score(truth, values)),
            'positive': int(truth.sum()),
            'negative': int(len(truth) - truth.sum()),
        }
    result = {
        'macro_auroc': float(np.mean([v['auroc'] for v in per.values()])),
        'macro_ap': float(np.mean([v['ap'] for v in per.values()])),
        'per_label': per,
        'checkpoint_sha256': checkpoint_hash,
    }
    rng = np.random.default_rng(42)
    bootstraps = []
    for _ in range(2000):
        sample = rng.integers(0, 500, size=500)
        values = []
        for j, finding in enumerate(findings):
            truth = targets[sample, j]
            if truth.min() == truth.max():
                break
            values.append(roc_auc_score(truth, study_scores[sample, LABELS.index(finding)]))
        if len(values) == len(findings):
            bootstraps.append(float(np.mean(values)))
    result['patient_bootstrap_95ci'] = np.quantile(bootstraps, [0.025, 0.975]).tolist()
    result['bootstrap_replicates'] = len(bootstraps)
    output = {
        'cohort': '500 CheXpert expert-labeled test studies; all available frontal images averaged per study',
        'groundtruth_sha256': manifest['sha256'],
        'frontal_images': len(paths),
        'studies': 500,
        'preprocessing': 'resize frontal image to 224x224 bilinear, RGB, ImageNet normalization; matches the cached validation images',
        'selected_model': selected,
        'selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
        'result': result,
        'limitation': 'Model selection used a patient-disjoint development set. Test performance is one held-out evaluation; published leaderboard preprocessing may differ.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({'selected_model': selected, 'macro_auroc': result['macro_auroc'],
                      'macro_ap': result['macro_ap']}, indent=2))


if __name__ == '__main__':
    main()
