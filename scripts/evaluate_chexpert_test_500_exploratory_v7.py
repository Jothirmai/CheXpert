"""Exploratory repeat test of a CNN after a prior test look."""

import argparse
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
from scripts.evaluate_chexpert_test_500 import DATA, LABELS, TestImages, locate_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=Path('checkpoints/chexpert_plus_findings_320_v7'))
    parser.add_argument('--image-size', type=int, default=320)
    args = parser.parse_args()
    out = args.checkpoint_dir
    if args.image_size not in (224, 320):
        raise RuntimeError('Unsupported image size')
    protocol = json.loads((out / 'protocol.json').read_text())
    if not protocol['test_result_known_before_experiment']:
        raise RuntimeError('This evaluator is only for a clearly labeled exploratory repeat test')
    if json.loads((out / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Model is incomplete')
    labels_manifest = json.loads((DATA / 'manifest.json').read_text())
    images_manifest = json.loads((DATA / 'image_manifest.json').read_text())
    raw = (DATA / 'groundtruth.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != labels_manifest['sha256']:
        raise RuntimeError('Expert test labels changed')
    if images_manifest['images'] != 668 or images_manifest['groundtruth_sha256'] != labels_manifest['sha256']:
        raise RuntimeError('Test image cohort changed')
    studies = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(studies) != 500:
        raise RuntimeError('Expected 500 expert-labeled studies')
    paths = []
    by_study = defaultdict(list)
    for index, row in enumerate(studies):
        for path in locate_study(DATA / 'images', row['Study']):
            by_study[index].append(len(paths))
            paths.append(path)
    if len(paths) != 518:
        raise RuntimeError(f'Expected 518 test frontal images; found {len(paths)}')
    dataset = TestImages(paths)
    dataset.transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
    torch.set_num_threads(4)
    checkpoint = out / 'best_model.pth'
    architecture = protocol.get('architecture', 'densenet201')
    if architecture == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None, num_classes=14)
    elif architecture == 'convnext_tiny':
        model = models.convnext_tiny(weights=None, num_classes=14)
    elif architecture == 'densenet201':
        model = models.densenet201(weights=None, num_classes=14)
    else:
        raise RuntimeError(f'Unsupported architecture: {architecture}')
    model = model.cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    chunks = []
    with torch.inference_mode():
        for images in loader:
            chunks.append(model(images.cuda(non_blocking=True)).sigmoid().cpu().numpy())
    image_scores = np.concatenate(chunks)
    scores = np.stack([image_scores[by_study[index]].mean(axis=0) for index in range(500)])
    findings = labels_manifest['findings']
    truth = np.array([[int(row[name]) for name in findings] for row in studies], dtype=int)
    per = {}
    for j, name in enumerate(findings):
        values = scores[:, LABELS.index(name)]
        per[name] = {'auroc': float(roc_auc_score(truth[:, j], values)),
                     'ap': float(average_precision_score(truth[:, j], values)),
                     'positive': int(truth[:, j].sum()),
                     'negative': int(500 - truth[:, j].sum())}
    macro = float(np.mean([v['auroc'] for v in per.values()]))
    rng = np.random.default_rng(44)
    bootstraps = []
    for _ in range(2000):
        sample = rng.integers(0, 500, size=500)
        values = []
        for j, name in enumerate(findings):
            yy = truth[sample, j]
            if yy.min() == yy.max():
                break
            values.append(roc_auc_score(yy, scores[sample, LABELS.index(name)]))
        if len(values) == len(findings):
            bootstraps.append(float(np.mean(values)))
    result = {
        'status': 'exploratory repeat test; earlier official test result was known before this experiment',
        'cohort': 'same 500 CheXpert expert-labeled test studies and 518 frontal images as the prior evaluation',
        'image_size': args.image_size,
        'architecture': architecture,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'macro_auroc': macro,
        'macro_ap': float(np.mean([v['ap'] for v in per.values()])),
        'per_label': per,
        'patient_bootstrap_95ci': np.quantile(bootstraps, [0.025, 0.975]).tolist(),
        'prior_v6_test_macro_auroc': json.loads((DATA / 'evaluation.json').read_text())['result']['macro_auroc'],
        'warning': 'This result cannot support a fresh leaderboard or state-of-the-art claim because the test set has been reused.',
    }
    (out / 'test_500_exploratory.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'macro_auroc': macro,
                      'patient_bootstrap_95ci': result['patient_bootstrap_95ci']}, indent=2))


if __name__ == '__main__':
    main()
