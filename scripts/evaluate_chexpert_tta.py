"""Exploratory fixed flip-TTA and equal-weight XRV ensemble on expert validation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_chexpert_expert_valid import ROOT, XRVImages, metrics, xrv_scores
from scripts.train_chexpert_plus_impression import Rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    expert = [json.loads(line) for line in (ROOT / 'expert_valid.jsonl').read_text().splitlines()]
    names = json.loads((ROOT / 'manifest.json').read_text())['labels']
    findings = json.loads((ROOT / 'expert_valid_manifest.json').read_text())['findings']
    j = {name: names.index(name) for name in findings}
    rows = Rows(ROOT / 'valid.jsonl')
    torch.set_num_threads(4)
    model = models.densenet121(weights=None, num_classes=14).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    original = []
    flipped = []
    with torch.inference_mode():
        for x, _, _ in DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True):
            x = x.cuda(non_blocking=True)
            original.append(model(x).sigmoid().cpu().numpy())
            flipped.append(model(x.flip(-1)).sigmoid().cpu().numpy())
    original = np.concatenate(original)
    tta = 0.5 * (original + np.concatenate(flipped))
    xmodel = xrv.models.DenseNet(weights='densenet121-res224-all').cuda()
    xscore = xrv_scores(xmodel, DataLoader(XRVImages(rows.rows), batch_size=32, num_workers=4, pin_memory=True))
    xaligned = np.zeros_like(original)
    for name, index in j.items():
        xaligned[:, index] = xscore[:, xmodel.pathologies.index('Effusion' if name == 'Pleural Effusion' else name)]
    blend = 0.5 * (tta + xaligned)
    result = {
        'cohort': 'CheXpert expert validation 202 frontal images; exploratory model configuration',
        'checkpoint': args.checkpoint,
        'single': metrics(original, expert, j),
        'horizontal_flip_tta': metrics(tta, expert, j),
        'fixed_50_50_tta_xrv_all': metrics(blend, expert, j),
        'publication_status': 'validation-only; requires independent expert test to claim generalization',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({name: {'macro_auroc': result[name]['macro_auroc'], 'macro_ap': result[name]['macro_ap']} for name in ['single', 'horizontal_flip_tta', 'fixed_50_50_tta_xrv_all']}, indent=2))


if __name__ == '__main__':
    main()
