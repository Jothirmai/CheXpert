"""Compute the selected ensemble's genuine 14-label development AUROC/AP."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_plus_uncertainty_v4 import Rows as Rows224
from scripts.train_chexpert_findings_320_v7 import Rows as Rows320

ROOT = Path('data/chexpert_plus_cnn_impression_v1')
HIGH = Path('data/chexpert_plus_cnn_impression_320_v1')
V6 = Path('checkpoints/chexpert_plus_densenet201_v6/best_model.pth')
V12 = Path('checkpoints/chexpert_plus_impression_320_v12/best_model.pth')
OUT = Path('reports/selected_ensemble_all14_development.json')
PROGRESS = Path('reports/selected_ensemble_all14_progress.json')
FIVE = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion']


def model(path):
    net = models.densenet201(weights=None, num_classes=14)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return net.cuda().eval()


def main():
    if OUT.exists():
        raise FileExistsError('A frozen 14-label result already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    source = Rows224(ROOT / 'development.jsonl')
    high = Rows320(HIGH / 'development.jsonl')
    if [r['source_path'] for r in source.rows] != [r['source_path'] for r in high.rows]:
        raise RuntimeError('224/320 development image order mismatch')
    names = json.loads((ROOT / 'manifest.json').read_text())['labels']
    selected = json.loads(Path('checkpoints/active_models.json').read_text())['cnn']
    if selected.get('architecture') != 'densenet201_v6_v12_tta':
        raise RuntimeError('Selected CNN changed')
    hashes = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (V6,V12)}
    if hashes != selected['sha256']:
        raise RuntimeError('Checkpoint hashes changed')
    batches224 = DataLoader(source, batch_size=16, num_workers=4, pin_memory=True)
    batches320 = DataLoader(high, batch_size=16, num_workers=4, pin_memory=True)
    v6, v12 = model(V6), model(V12)
    scores, targets, masks = [], [], []
    with torch.inference_mode():
        for index, ((x224,y,m),(x320,y320,m320)) in enumerate(zip(batches224,batches320),1):
            if not torch.equal(y,y320) or not torch.equal(m,m320):
                raise RuntimeError('224/320 labels differ')
            x224=x224.cuda(non_blocking=True)
            x320=x320.cuda(non_blocking=True)
            pred=(.5*v6(x224).sigmoid()+.25*v12(x320).sigmoid()
                  +.25*v12(x320.flip(-1)).sigmoid())
            scores.append(pred.cpu().numpy())
            targets.append(y.numpy())
            masks.append(m.numpy())
            if index%100==0:
                PROGRESS.write_text(json.dumps({'phase':'scoring','batches':index,
                    'total_batches':len(batches224),'fraction':index/len(batches224)}))
    scores=np.concatenate(scores)
    targets=np.concatenate(targets)
    masks=np.concatenate(masks).astype(bool)
    per={}
    for j,name in enumerate(names):
        keep=masks[:,j]
        truth=targets[keep,j].astype(int)
        values=scores[keep,j]
        if len(np.unique(truth))!=2:
            per[name]={'positive':int(truth.sum()),'negative':int(len(truth)-truth.sum()),
                       'auroc':None,'ap':None}
        else:
            per[name]={'positive':int(truth.sum()),'negative':int(len(truth)-truth.sum()),
                       'auroc':float(roc_auc_score(truth,values)),
                       'ap':float(average_precision_score(truth,values))}
    valid=[item for item in per.values() if item['auroc'] is not None]
    result={'dataset':'CheXpert Plus Impression report-derived labels',
            'cohort':'frozen patient-disjoint development split',
            'images':len(scores),'evaluable_labels':len(valid),
            'macro_auroc':float(np.mean([x['auroc'] for x in valid])),
            'macro_ap':float(np.mean([x['ap'] for x in valid])),
            'five_finding_macro_auroc':float(np.mean([per[x]['auroc'] for x in FIVE])),
            'five_finding_macro_ap':float(np.mean([per[x]['ap'] for x in FIVE])),
            'per_label':per,'checkpoint_sha256':hashes,
            'limitations':'Report-derived weak labels and inspected development cohort; not the five-finding expert-labeled test score.'}
    previous=json.loads(Path('reports/chexpert_densenet201_v12_tta_blends.json').read_text())
    expected=previous['development']['equal_v6_v12_tta']['macro_auroc']
    if abs(result['five_finding_macro_auroc']-expected)>1e-6:
        raise RuntimeError(f'Five-finding result differs from frozen selection: {result["five_finding_macro_auroc"]} vs {expected}')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    PROGRESS.write_text(json.dumps({'phase':'complete','result':str(OUT)}))
    print(json.dumps({k:result[k] for k in ['images','evaluable_labels','macro_auroc','macro_ap','five_finding_macro_auroc']}),flush=True)


if __name__=='__main__':
    main()
