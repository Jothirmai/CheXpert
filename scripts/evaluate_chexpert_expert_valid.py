"""Compare CNN checkpoints on the same 202 expert-labeled frontal validation images."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_plus_impression import ROOT, Rows

OUT = Path('checkpoints/chexpert_plus_impression_v3')


class XRVImages(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]['image_path']) as image:
            pixels = np.asarray(image.convert('L'), dtype=np.float32)
        return torch.from_numpy(xrv.datasets.normalize(pixels, maxval=255)[None, :, :].astype(np.float32))


def cnn_scores(model, loader):
    model.eval()
    output = []
    with torch.inference_mode():
        for x, _, _ in loader:
            output.append(model(x.cuda(non_blocking=True)).sigmoid().cpu().numpy())
    return np.concatenate(output)


def xrv_scores(model, loader):
    model.eval()
    output = []
    with torch.inference_mode():
        for x in loader:
            output.append(model(x.cuda(non_blocking=True)).cpu().numpy())
    return np.concatenate(output)


def metrics(score, labels, finding_indices):
    per = {}
    for finding, j in finding_indices.items():
        truth = np.asarray([row['labels'][finding] for row in labels])
        keep = np.isin(truth, [0, 1])
        yy = truth[keep].astype(int)
        ss = score[keep, j]
        per[finding] = {'positive': int(yy.sum()), 'negative': int(len(yy) - yy.sum()), 'auroc': float(roc_auc_score(yy, ss)), 'ap': float(average_precision_score(yy, ss))}
    return {'macro_auroc': float(np.mean([entry['auroc'] for entry in per.values()])), 'macro_ap': float(np.mean([entry['ap'] for entry in per.values()])), 'per_label': per}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--candidate-out', default=str(OUT))
    args = parser.parse_args()
    candidate_out = Path(args.candidate_out)
    manifest = json.loads((ROOT / 'expert_valid_manifest.json').read_text())
    path = ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed since freezing')
    labels = [json.loads(line) for line in path.read_text().splitlines()]
    rows = Rows(ROOT / 'valid.jsonl')
    if [row['source_path'] for row in labels] != [row['source_path'] for row in rows.rows]:
        raise RuntimeError('Expert labels and images are out of order')
    names = json.loads((ROOT / 'manifest.json').read_text())['labels']
    finding_indices = {finding: names.index(finding) for finding in manifest['findings']}
    torch.set_num_threads(4)
    loader = DataLoader(rows, batch_size=32, num_workers=4, pin_memory=True)
    model = models.densenet121(weights=None, num_classes=14).cuda()
    results = {}
    checkpoints = {
        'hct_stage1': '/home/doli/HCT_LSR/hct14_ensemble/models/densenet121_hct_lsr_14_stage1_best.pt',
        'hct_stage2': '/home/doli/HCT_LSR/hct14_ensemble/models/densenet121_hct_lsr_14_stage2_best.pt',
    }
    if args.candidate:
        progress = json.loads((candidate_out / 'progress.json').read_text())
        if progress.get('phase') != 'complete':
            raise RuntimeError('Candidate not fully trained')
        protocol = json.loads((candidate_out / 'protocol.json').read_text())
        training_root = Path('data/chexpert_plus_cnn_findings_subset_v1') if protocol['label_file'] == 'findings_fixed.json' else ROOT
        for split, digest in protocol['data_hashes'].items():
            if hashlib.sha256((training_root / f'{split}.jsonl').read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen {split} training data changed')
        if candidate_out != OUT:
            checkpoints['ours_impression_v3'] = str(OUT / 'best_model.pth')
        if candidate_out.name == 'chexpert_plus_findings_v5':
            checkpoints['ours_uones_v4'] = 'checkpoints/chexpert_plus_uncertainty_v4/best_model.pth'
        if candidate_out.name == 'chexpert_plus_pairrank_v8':
            checkpoints['ours_densenet201_v6'] = 'checkpoints/chexpert_plus_densenet201_v6/best_model.pth'
        candidate_name = (
            'ours_impression' if candidate_out == OUT else
            'ours_findings_v5' if candidate_out.name == 'chexpert_plus_findings_v5' else
            'ours_densenet201_v6' if candidate_out.name == 'chexpert_plus_densenet201_v6' else
            'ours_pairrank_v8' if candidate_out.name == 'chexpert_plus_pairrank_v8' else
            'ours_uones_v4'
        )
        checkpoints[candidate_name] = str(candidate_out / 'best_model.pth')
    for name, checkpoint in checkpoints.items():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        candidate_model = (models.densenet201(weights=None, num_classes=14).cuda()
                           if name in ('ours_densenet201_v6', 'ours_pairrank_v8') else model)
        candidate_model.load_state_dict(state['model_state_dict'] if name == 'hct_stage1' else state)
        results[name] = metrics(cnn_scores(candidate_model, loader), labels, finding_indices)
        if candidate_model is not model:
            del candidate_model
    xloader = DataLoader(XRVImages(rows.rows), batch_size=32, num_workers=4, pin_memory=True)
    for weights in ('densenet121-res224-chex', 'densenet121-res224-all'):
        xmodel = xrv.models.DenseNet(weights=weights).cuda()
        indices = {finding: xmodel.pathologies.index('Effusion' if finding == 'Pleural Effusion' else finding) for finding in manifest['findings']}
        results['xrv_' + weights.rsplit('-', 1)[-1]] = metrics(xrv_scores(xmodel, xloader), labels, indices)
        del xmodel
    result = {
        'cohort': '202 frontal images and 200 patients from CheXpert original expert-labeled validation split',
        'source_repo': manifest['source_repo'],
        'source_revision': manifest['source_revision'],
        'label_sha256': manifest['jsonl_sha256'],
        'findings': manifest['findings'],
        'results': results,
        'candidate_selected_by': (
            'patient-disjoint CheXpert Plus findings-labeled development split'
            if args.candidate and protocol['label_file'] == 'findings_fixed.json'
            else 'patient-disjoint CheXpert Plus impression-labeled development split'
            if args.candidate else None
        ),
        'candidate_checkpoint': str(candidate_out / 'best_model.pth') if args.candidate else None,
        'limitations': 'This is the 202 frontal-image subset of the 234-image expert validation set, not the 500-study CheXpert challenge test set. Published challenge test scores are not directly rankable against it.',
    }
    candidate_out.mkdir(parents=True, exist_ok=True)
    (candidate_out / 'expert_valid_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({name: {key: value[key] for key in ('macro_auroc', 'macro_ap')} for name, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
