"""Cache frozen v12 DenseNet201 features for category-wise head tuning."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_chexpert_findings_320_v7 import Rows


DATA = Path('data/chexpert_plus_cnn_impression_320_v1')
SOURCE = Path('checkpoints/chexpert_plus_impression_320_v12/best_model.pth')
OUT = Path('checkpoints/chexpert_plus_category_heads_v15')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for split, digest in manifest['sha256'].items():
        if hashlib.sha256((DATA / f'{split}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen {split} data changed')
    if json.loads((SOURCE.parent / 'progress.json').read_text())['phase'] != 'complete':
        raise RuntimeError('Source checkpoint incomplete')
    torch.set_num_threads(4)
    model = models.densenet201(weights=None, num_classes=14).cuda().eval()
    model.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=True))
    start = time.monotonic()
    for split in ('fit', 'development', 'valid'):
        dataset = Rows(DATA / f'{split}.jsonl')
        loader = DataLoader(dataset, batch_size=24, num_workers=8, pin_memory=True)
        temporary = OUT / f'{split}_features.partial.npy'
        completed = OUT / f'{split}_features.npy'
        if completed.exists():
            print('EXISTS', split, len(dataset), flush=True)
            continue
        features = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                             shape=(len(dataset), 1920))
        index = 0
        with torch.inference_mode():
            for step, (images, _, _) in enumerate(loader, 1):
                images = images.cuda(non_blocking=True)
                activation = torch.nn.functional.relu(model.features(images), inplace=False)
                pooled = torch.nn.functional.adaptive_avg_pool2d(activation, (1, 1)).flatten(1)
                if step == 1:
                    direct = model(images)
                    reconstructed = model.classifier(pooled)
                    if not torch.allclose(direct, reconstructed, atol=2e-4, rtol=2e-4):
                        raise RuntimeError('Extracted features do not reproduce classifier logits')
                batch = pooled.cpu().numpy().astype(np.float16)
                features[index:index + len(batch)] = batch
                index += len(batch)
                if step % 100 == 0:
                    progress = {'phase': 'extracting', 'split': split, 'rows': index,
                                'total_rows': len(dataset), 'fraction': index / len(dataset),
                                'elapsed_s': time.monotonic() - start}
                    (OUT / 'feature_progress.json').write_text(json.dumps(progress))
                    print('PROGRESS', json.dumps(progress), flush=True)
        if index != len(dataset):
            raise RuntimeError(f'Incomplete feature cache for {split}')
        features.flush()
        del features
        temporary.rename(completed)
        labels = np.array([[np.nan if value is None else value for value in row['labels']]
                           for row in dataset.rows], dtype=np.float32)
        np.save(OUT / f'{split}_raw_labels.npy', labels)
        print('EXTRACTED', split, index, flush=True)
    protocol = {
        'source_checkpoint_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'data_hashes': manifest['sha256'],
        'feature_dimension': 1920,
        'splits': {name: manifest['counts'][name]['images'] for name in ('fit', 'development', 'valid')},
        'test_result_known_before_experiment': True,
        'test_used_for_feature_extraction': False,
    }
    (OUT / 'feature_protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUT / 'feature_progress.json').write_text(json.dumps({'phase': 'complete', 'elapsed_s': time.monotonic() - start}))
    print('FEATURE_EXTRACTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
