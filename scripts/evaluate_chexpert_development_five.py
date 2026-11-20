"""Score a frozen CNN on the common patient-disjoint impression development split."""

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
from scripts.train_chexpert_plus_uncertainty_v4 import ROOT, Rows, evaluate


FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.checkpoint_dir
    protocol = json.loads((directory / 'protocol.json').read_text())
    if json.loads((directory / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Checkpoint is incomplete')
    development_path = ROOT / 'development.jsonl'
    digest = json.loads((ROOT / 'development_split.json').read_text())['sha256']['development']
    if hashlib.sha256(development_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen development data changed')
    torch.set_num_threads(4)
    model = (models.densenet201(weights=None, num_classes=14)
             if 'DenseNet201' in protocol['model'] else models.densenet121(weights=None, num_classes=14))
    model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True))
    model.cuda()
    loader = DataLoader(Rows(development_path), batch_size=32, num_workers=4, pin_memory=True)
    fourteen = evaluate(model, loader, 'cuda:0')
    five = {name: fourteen['per_label'][name] for name in FINDINGS}
    result = {
        'cohort': '19,688 patient-disjoint CheXpert Plus train images; impression-derived labels',
        'label_policy': 'blank -> 0; explicit -1 masked',
        'findings': FINDINGS,
        'macro_auroc': float(np.mean([item['auroc'] for item in five.values()])),
        'macro_ap': float(np.mean([item['ap'] for item in five.values()])),
        'per_label': five,
        'checkpoint_sha256': hashlib.sha256((directory / 'best_model.pth').read_bytes()).hexdigest(),
    }
    (directory / 'impression_development_five.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'macro_auroc': result['macro_auroc'], 'macro_ap': result['macro_ap']}))


if __name__ == '__main__':
    main()
