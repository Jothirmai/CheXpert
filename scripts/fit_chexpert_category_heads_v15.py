"""Fine-tune five frozen-backbone classifier rows using explicit labels only.

This is a local category-wise head experiment inspired by CFT, not a claim to
reproduce the published CFT genetic-algorithm implementation.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path('data/chexpert_plus_cnn_impression_320_v1')
EXPERT_ROOT = Path('data/chexpert_plus_cnn_impression_v1')
SOURCE = Path('checkpoints/chexpert_plus_impression_320_v12/best_model.pth')
OUT = Path('checkpoints/chexpert_plus_category_heads_v15')
NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
PENALTIES = (0.01, 0.1, 1.0)


def metric(truth, scores):
    per = {}
    for j, name in enumerate(NAMES):
        keep = np.isin(truth[:, j], [0, 1])
        yy = truth[keep, j].astype(int)
        ss = scores[keep, j]
        per[name] = {
            'positive': int(yy.sum()), 'negative': int(len(yy) - yy.sum()),
            'auroc': float(roc_auc_score(yy, ss)),
            'ap': float(average_precision_score(yy, ss)),
        }
    return {
        'macro_auroc': float(np.mean([entry['auroc'] for entry in per.values()])),
        'macro_ap': float(np.mean([entry['ap'] for entry in per.values()])),
        'per_label': per,
    }


def main():
    torch.set_num_threads(4)
    torch.manual_seed(50)
    protocol = json.loads((OUT / 'feature_protocol.json').read_text())
    if json.loads((OUT / 'feature_progress.json').read_text())['phase'] != 'complete':
        raise RuntimeError('Feature extraction is incomplete')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != protocol['source_checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for split, digest in protocol['data_hashes'].items():
        if hashlib.sha256((ROOT / f'{split}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen {split} data changed')
    names = json.loads((EXPERT_ROOT / 'manifest.json').read_text())['labels']
    indices = [names.index(name) for name in NAMES]
    fit_raw = np.load(OUT / 'fit_raw_labels.npy')[:, indices]
    dev_raw = np.load(OUT / 'development_raw_labels.npy')[:, indices]
    dev_truth = np.nan_to_num(dev_raw, nan=0)
    device = 'cuda:0'
    fit_x = torch.from_numpy(np.load(OUT / 'fit_features.npy').astype(np.float32)).to(device)
    dev_x = torch.from_numpy(np.load(OUT / 'development_features.npy').astype(np.float32)).to(device)
    expert_x = torch.from_numpy(np.load(OUT / 'valid_features.npy').astype(np.float32)).to(device)
    state = torch.load(SOURCE, map_location='cpu', weights_only=True)
    source_w = state['classifier.weight'][indices].to(device)
    source_b = state['classifier.bias'][indices].to(device)
    with torch.inference_mode():
        baseline_dev = torch.sigmoid(torch.nn.functional.linear(dev_x, source_w, source_b)).cpu().numpy()
    trial = {}
    selected_w = source_w.clone()
    selected_b = source_b.clone()
    started = time.monotonic()
    for j, name in enumerate(NAMES):
        known = np.isin(fit_raw[:, j], [0, 1])
        positive = int((fit_raw[:, j] == 1).sum())
        negative = int((fit_raw[:, j] == 0).sum())
        if positive < 50 or negative < 50:
            raise RuntimeError(f'Too few explicit labels for {name}')
        x = fit_x[torch.from_numpy(np.flatnonzero(known)).to(device)]
        y = torch.from_numpy(fit_raw[known, j]).to(device)
        positives = y == 1
        negatives = y == 0
        dev_keep = np.isin(dev_truth[:, j], [0, 1])
        best = float(roc_auc_score(dev_truth[dev_keep, j], baseline_dev[dev_keep, j]))
        trial[name] = {'explicit_fit_positive': positive, 'explicit_fit_negative': negative,
                       'baseline_development_auroc': best, 'candidates': []}
        for penalty in PENALTIES:
            delta_w = torch.nn.Parameter(torch.zeros_like(source_w[j]))
            delta_b = torch.nn.Parameter(torch.zeros_like(source_b[j]))
            optimizer = torch.optim.Adam([delta_w, delta_b], lr=0.003)
            for _ in range(120):
                logits = torch.nn.functional.linear(x, (source_w[j] + delta_w)[None],
                                                    (source_b[j] + delta_b)[None]).squeeze(1)
                loss_pos = torch.nn.functional.softplus(-logits[positives]).mean()
                loss_neg = torch.nn.functional.softplus(logits[negatives]).mean()
                loss = (loss_pos + loss_neg) / 2 + penalty * delta_w.square().sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.inference_mode():
                candidate_w = source_w[j] + delta_w
                candidate_b = source_b[j] + delta_b
                score = torch.sigmoid(torch.nn.functional.linear(dev_x, candidate_w[None],
                                                                 candidate_b[None])).squeeze(1).cpu().numpy()
            auc = float(roc_auc_score(dev_truth[dev_keep, j], score[dev_keep]))
            trial[name]['candidates'].append({'penalty': penalty, 'development_auroc': auc})
            if auc > best:
                best = auc
                selected_w[j] = candidate_w.detach()
                selected_b[j] = candidate_b.detach()
                trial[name]['selected_penalty'] = penalty
        trial[name]['selected_development_auroc'] = best
        progress = {'phase': 'fitting_heads', 'completed_classes': j + 1, 'total_classes': len(NAMES),
                    'last_class': name, 'elapsed_s': time.monotonic() - started}
        (OUT / 'progress.json').write_text(json.dumps(progress))
        (OUT / 'head_trials.json').write_text(json.dumps(trial, indent=2))
        print('CLASS', name, json.dumps(trial[name]), flush=True)
        del x, y
    with torch.inference_mode():
        candidate_dev = torch.sigmoid(torch.nn.functional.linear(dev_x, selected_w, selected_b)).cpu().numpy()
        baseline_expert = torch.sigmoid(torch.nn.functional.linear(expert_x, source_w, source_b)).cpu().numpy()
        candidate_expert = torch.sigmoid(torch.nn.functional.linear(expert_x, selected_w, selected_b)).cpu().numpy()
    expert_manifest = json.loads((EXPERT_ROOT / 'expert_valid_manifest.json').read_text())
    expert_path = EXPERT_ROOT / 'expert_valid.jsonl'
    if hashlib.sha256(expert_path.read_bytes()).hexdigest() != expert_manifest['jsonl_sha256']:
        raise RuntimeError('Expert labels changed')
    expert_rows = [json.loads(line) for line in expert_path.read_text().splitlines()]
    high_rows = [json.loads(line) for line in (ROOT / 'valid.jsonl').read_text().splitlines()]
    if [row['source_path'] for row in expert_rows] != [row['source_path'] for row in high_rows]:
        raise RuntimeError('Expert image order mismatch')
    expert_truth = np.array([[row['labels'][name] for name in NAMES] for row in expert_rows], dtype=int)
    result = {
        'method': 'Frozen DenseNet201 features; five classifier rows tuned independently on explicit 0/1 fit labels',
        'inspired_by': 'Category-wise Fine-Tuning (Chong et al.); this implementation does not use their genetic algorithm',
        'penalties': PENALTIES,
        'selection': 'Per-class highest AUROC on frozen patient-disjoint Impression development split, including unchanged baseline as candidate',
        'development_baseline': metric(dev_truth, baseline_dev),
        'development_candidate': metric(dev_truth, candidate_dev),
        'expert_validation_baseline_exploratory': metric(expert_truth, baseline_expert),
        'expert_validation_candidate_exploratory': metric(expert_truth, candidate_expert),
        'test_result_known_before_experiment': True,
    }
    (OUT / 'selection.json').write_text(json.dumps(result, indent=2))
    for j, index in enumerate(indices):
        state['classifier.weight'][index] = selected_w[j].cpu()
        state['classifier.bias'][index] = selected_b[j].cpu()
    torch.save(state, OUT / 'best_model.pth')
    training_protocol = {
        'architecture': 'densenet201',
        'model': 'DenseNet201 frozen backbone with five category-wise re-fitted classifier rows',
        'initialization': 'v12 DenseNet201 trained on CheXpert Plus; original backbone initialized from ImageNet-1K',
        'source_checkpoint_sha256': protocol['source_checkpoint_sha256'],
        'data_hashes': protocol['data_hashes'],
        'label_file': manifest['label_file'],
        'label_policy': 'Fine-tune classifier row on explicit 0/1 labels only; blanks and uncertain values ignored',
        'selection': result['selection'],
        'test_result_known_before_experiment': True,
    }
    (OUT / 'protocol.json').write_text(json.dumps(training_protocol, indent=2))
    (OUT / 'progress.json').write_text(json.dumps({'phase': 'complete', 'elapsed_s': time.monotonic() - started}))
    print('COMPLETE', json.dumps({
        'development_baseline': result['development_baseline']['macro_auroc'],
        'development_candidate': result['development_candidate']['macro_auroc'],
        'expert_baseline': result['expert_validation_baseline_exploratory']['macro_auroc'],
        'expert_candidate': result['expert_validation_candidate_exploratory']['macro_auroc'],
    }), flush=True)


if __name__ == '__main__':
    main()
