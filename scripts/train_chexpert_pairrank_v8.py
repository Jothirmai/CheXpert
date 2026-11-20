"""Fine-tune the CNN with BCE plus five-finding pairwise ranking loss."""

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
from scripts.train_chexpert_plus_uncertainty_v4 import ROOT, Rows, TrainRows, evaluate


SOURCE = Path('checkpoints/chexpert_plus_densenet201_v6/best_model.pth')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def pairwise_rank_loss(logits, targets, masks, indices):
    values = []
    for j in indices:
        positive = logits[(targets[:, j] == 1) & masks[:, j], j]
        negative = logits[(targets[:, j] == 0) & masks[:, j], j]
        if positive.numel() and negative.numel():
            values.append(torch.nn.functional.softplus(-(positive[:, None] - negative[None, :])).mean())
    if not values:
        return logits.sum() * 0
    return torch.stack(values).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('checkpoints/chexpert_plus_pairrank_v8'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Use a fresh output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if json.loads((SOURCE.parent / 'progress.json').read_text()).get('phase') != 'complete':
        raise RuntimeError('Source model is incomplete')
    torch.manual_seed(45)
    np.random.seed(45)
    random.seed(45)
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    split = json.loads((ROOT / 'development_split.json').read_text())
    for name, digest in split['sha256'].items():
        if hashlib.sha256((ROOT / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen split changed: {name}')
    indices = [manifest['labels'].index(name) for name in FINDINGS]
    fit = DataLoader(TrainRows(ROOT / 'fit.jsonl'), batch_size=args.batch_size, shuffle=True,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    dev = DataLoader(Rows(ROOT / 'development.jsonl'), batch_size=32,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = models.densenet201(weights=None, num_classes=14)
    model.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=True))
    model.cuda()
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
        'data_hashes': split['sha256'],
        'label_file': 'impression_fixed.json',
        'label_file_sha256': manifest['source_label_sha256'],
        'model': 'torchvision DenseNet201, full-parameter fine-tuning from v6',
        'source_checkpoint_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'input': '224x224 frontal radiographs',
        'fit_images': split['counts']['fit']['images'],
        'development_images': split['counts']['development']['images'],
        'label_policy': 'blank -> 0; uncertain Atelectasis and Edema -> 1 in fit; other uncertain -> masked',
        'development_label_policy': 'blank -> 0; explicit -1 masked',
        'loss': '14-label weighted BCE plus 0.2 times mean five-finding pairwise softplus ranking loss within each fit batch',
        'positive_weights': positive_weights.cpu().tolist(),
        'augmentation': 'random horizontal flip probability 0.5 on fit only',
        'optimizer': 'AdamW lr 1e-5 weight_decay 1e-4, cosine decay per epoch, mixed precision',
        'selection': 'five-finding macro AUROC on patient-disjoint impression-labeled development split',
        'selection_findings': FINDINGS,
        'test_labels_not_used_for_optimization': True,
        'test_result_known_before_experiment': True,
        'evaluation_status': 'exploratory; a new independent test is required for a fresh leaderboard claim',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': 45,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=3e-6)
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
                bce = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target, pos_weight=positive_weights, reduction='none') * mask
                ).sum() / mask.sum().clamp_min(1)
                rank = pairwise_rank_loss(logits.float(), target, mask, indices)
                loss = bce + 0.2 * rank
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
