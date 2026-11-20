"""Compare EfficientNetV2-S and ConvNeXt Tiny on one frozen CXR split."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_findings_320_v7 import ROOT, Rows, TrainRows
from scripts.train_chexpert_plus_uncertainty_v4 import evaluate


FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
OPTIONS = {
    'efficientnet_v2_s': {'output': 'checkpoints/chexpert_plus_efficientnet_v2_s_v9', 'lr': 1e-4, 'seed': 46},
    'convnext_tiny': {'output': 'checkpoints/chexpert_plus_convnext_tiny_v10', 'lr': 5e-5, 'seed': 47},
}


def initialize(architecture):
    if architecture == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 14)
    elif architecture == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 14)
    else:
        raise ValueError(architecture)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=OPTIONS, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    option = dict(OPTIONS[args.architecture])
    if args.learning_rate is not None:
        option['lr'] = args.learning_rate
    if args.seed is not None:
        option['seed'] = args.seed
    data_root = args.data_root
    out = args.output or Path(option['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Use a fresh output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.manual_seed(option['seed'])
    np.random.seed(option['seed'])
    random.seed(option['seed'])
    torch.set_num_threads(4)
    manifest = json.loads((data_root / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((data_root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen data changed: {name}')
    fit = DataLoader(TrainRows(data_root / 'fit.jsonl'), batch_size=args.batch_size, shuffle=True,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    dev = DataLoader(Rows(data_root / 'development.jsonl'), batch_size=24,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = initialize(args.architecture).cuda()
    raw = np.array([[np.nan if value is None else value for value in row['labels']]
                    for row in fit.dataset.rows], dtype=np.float32)
    for j in (7, 4):
        raw[raw[:, j] == -1, j] = 1
    known = raw != -1
    positive = (raw == 1) & known
    pos = positive.sum(axis=0)
    neg = (known & ~positive).sum(axis=0)
    positive_weights = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 5),
                                    dtype=torch.float32, device='cuda:0')
    protocol = {
        'data_hashes': manifest['sha256'],
        'data_root': str(data_root),
        'label_file': manifest['label_file'],
        'label_file_sha256': manifest['label_source_sha256'],
        'architecture': args.architecture,
        'model': f'torchvision {args.architecture}, all parameters fine-tuned on CheXpert Plus {manifest["label_file"]} fit split',
        'initialization': f'ImageNet-1K pretrained torchvision {args.architecture} backbone; random 14-label classification head',
        'input': '320x320 bilinear-resized CheXpert-v1.0-small frontal radiographs',
        'fit_images': manifest['counts']['fit']['images'],
        'development_images': manifest['counts']['development']['images'],
        'label_policy': 'blank -> 0; uncertain Atelectasis and Edema -> 1 in fit; other uncertain -> masked',
        'development_label_policy': 'blank -> 0; explicit -1 masked',
        'augmentation': 'horizontal flip probability 0.5 on fit only',
        'optimizer': f'AdamW lr {option["lr"]} weight_decay 1e-4, cosine decay per epoch, mixed precision',
        'positive_weights': positive_weights.cpu().tolist(),
        'selection': f'five-finding mean AUROC on the patient-disjoint {manifest["label_file"]} development split',
        'selection_findings': FINDINGS,
        'test_labels_not_used_for_optimization': True,
        'test_result_known_before_experiment': True,
        'evaluation_status': 'exploratory; new independent test needed for a fresh leaderboard claim',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': option['seed'],
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=option['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    best = -1.0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (image, target, mask) in enumerate(fit, 1):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(image)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target, pos_weight=positive_weights, reduction='none') * mask
                ).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss)
            if step % 100 == 0:
                progress = {'phase': 'training', 'epoch': epoch, 'epochs': args.epochs,
                            'batch': step, 'batches_per_epoch': len(fit),
                            'fraction': ((epoch - 1) + step / len(fit)) / args.epochs,
                            'elapsed_s': time.monotonic() - started}
                (out / 'progress.json').write_text(json.dumps(progress))
                print('PROGRESS', json.dumps(progress), flush=True)
        measure = evaluate(model, dev, 'cuda:0')
        five_auc = float(np.mean([measure['per_label'][name]['auroc'] for name in FINDINGS]))
        five_ap = float(np.mean([measure['per_label'][name]['ap'] for name in FINDINGS]))
        entry = {'epoch': epoch, 'train_loss': total_loss / len(fit), 'development': measure,
                 'five_finding_macro_auroc': five_auc, 'five_finding_macro_ap': five_ap}
        history.append(entry)
        print('EPOCH', json.dumps({'epoch': epoch, 'five_auroc': five_auc,
                                   'five_ap': five_ap, 'fourteen_auroc': measure['macro_auroc']}), flush=True)
        if five_auc > best:
            best = five_auc
            torch.save(model.state_dict(), out / 'best_model.pth')
            (out / 'best_metrics.json').write_text(json.dumps(entry, indent=2))
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        scheduler.step()
    (out / 'progress.json').write_text(json.dumps({'phase': 'complete', 'epochs': args.epochs,
                                                    'elapsed_s': time.monotonic() - started}))
    print('COMPLETE', best, flush=True)


if __name__ == '__main__':
    main()
