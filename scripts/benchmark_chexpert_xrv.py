"""Evaluate released TorchXRayVision CheXpert weights on the frozen official validation rows."""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_plus_uignore import ROOT


class Images(Dataset):
    def __init__(self, split='valid', limit=None):
        self.rows = [json.loads(s) for s in (ROOT / f'{split}.jsonl').read_text().splitlines()]
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]['image_path']) as image:
            pixels = np.asarray(image.convert('L'), dtype=np.float32)
        normalized = xrv.datasets.normalize(pixels, maxval=255).astype(np.float32)
        return torch.from_numpy(normalized[None, :, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'valid'], default='valid')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--data-root', default='data/chexpert_plus_cnn_v1')
    parser.add_argument('--out')
    parser.add_argument('--weights', default='densenet121-res224-chex')
    args = parser.parse_args()
    global ROOT
    ROOT = Path(args.data_root)
    torch.set_num_threads(4)
    data = Images(args.split, args.limit)
    model = xrv.models.DenseNet(weights=args.weights).eval().cuda()
    predictions = []
    with torch.inference_mode():
        for images in DataLoader(data, batch_size=32, num_workers=4):
            predictions.append(model(images.cuda()).cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = json.loads((ROOT / 'manifest.json').read_text())['labels']
    mapping = {'Pleural Effusion': 'Effusion'}
    per_label = {}
    for j, label in enumerate(labels):
        source = mapping.get(label, label)
        if source not in model.pathologies:
            continue
        raw = np.array([np.nan if row['labels'][j] is None else row['labels'][j] for row in data.rows], dtype=float)
        keep = raw != -1
        y = (raw[keep] == 1).astype(int)
        score = predictions[keep, model.pathologies.index(source)]
        if y.sum() >= 3 and (len(y) - y.sum()) >= 10:
            per_label[label] = {'auroc': float(roc_auc_score(y, score)), 'ap': float(average_precision_score(y, score)), 'positive': int(y.sum()), 'negative': int(len(y)-y.sum())}
    result = {'model': f'TorchXRayVision {args.weights} released weights', 'cohort': f'CheXpert Plus {args.split} {len(data)} frontal images', 'label_policy': 'blank negative, uncertain ignored', 'macro_auroc': float(np.mean([v['auroc'] for v in per_label.values()])), 'macro_ap': float(np.mean([v['ap'] for v in per_label.values()])), 'per_label': per_label, 'limitations': 'Released model may have trained on CheXpert Plus train patients; this is a diagnostic if split=train, and is not a published leaderboard score.'}
    out = Path(args.out) if args.out else Path('checkpoints/chexpert_plus_uignore_v2/xrv_chex_baseline.json' if args.split == 'valid' and args.limit is None else 'checkpoints/chexpert_plus_uignore_v2/xrv_train_diagnostic.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'labels': len(per_label), 'macro_auroc': result['macro_auroc'], 'macro_ap': result['macro_ap']}))


if __name__ == '__main__':
    main()
