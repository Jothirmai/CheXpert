"""Materialize the existing patient-disjoint Impression split at 320 pixels."""

import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


REVISION = 'ca5a840e438b5d2d0fa07f560782f196b38a84aa'
SOURCE = Path.home() / '.cache/huggingface/hub/datasets--danjacobellis--chexpert/snapshots' / REVISION / 'data'
ORIGINAL = Path('data/chexpert_plus_cnn_impression_v1')
FINDINGS_320 = Path('data/chexpert_plus_cnn_findings_320_v1/images')
OUT = Path('data/chexpert_plus_cnn_impression_320_v1')


def load(name):
    return [json.loads(line) for line in (ORIGINAL / f'{name}.jsonl').read_text().splitlines()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('This cache is already frozen')
    original_split = json.loads((ORIGINAL / 'development_split.json').read_text())
    for name, digest in original_split['sha256'].items():
        if hashlib.sha256((ORIGINAL / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen source split changed: {name}')
    rows = {name: load(name) for name in ('fit', 'development', 'valid', 'expert_valid')}
    wanted = {row['source_path'] for name in ('fit', 'development', 'valid') for row in rows[name]}
    reused = 0
    for relative in wanted:
        prior = FINDINGS_320 / relative
        target = OUT / 'images' / relative
        if prior.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(prior, target)
            reused += 1
    print('REUSED_HIGH_RESOLUTION_IMAGES', reused, flush=True)
    missing = {relative for relative in wanted if not (OUT / 'images' / relative).is_file()}
    train_files = [SOURCE / f'train-{index:05d}-of-00023.parquet' for index in range(23)]
    valid_files = [SOURCE / 'validation-00000-of-00001.parquet']
    if not all(path.is_file() for path in train_files + valid_files):
        raise FileNotFoundError('Pinned CheXpert image snapshot is incomplete')
    for index, source in enumerate(train_files + valid_files, 1):
        parquet = pq.ParquetFile(source)
        for group in range(parquet.num_row_groups):
            batch = parquet.read_row_group(group, columns=['Path', 'image'])
            for item in batch.to_pylist():
                relative = item['Path'].removeprefix('CheXpert-v1.0-small/')
                if relative not in missing:
                    continue
                target = OUT / 'images' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item['image']['bytes'])
                missing.remove(relative)
        print('SOURCE', index, '/', len(train_files + valid_files), 'remaining', len(missing), flush=True)
    if missing:
        raise RuntimeError(f'{len(missing)} required high-resolution images unavailable')
    hashes = {}
    counts = {}
    for name, split in rows.items():
        for row in split:
            row['image_path'] = str((OUT / 'images' / row['source_path']).resolve())
        path = OUT / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in split))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[name] = {'images': len(split), 'patients': len({row['patient_id'] for row in split})}
    fit = {row['patient_id'] for row in rows['fit']}
    dev = {row['patient_id'] for row in rows['development']}
    valid = {row['patient_id'] for row in rows['valid']}
    if fit & dev or fit & valid or dev & valid:
        raise RuntimeError('Patient overlap among frozen splits')
    manifest = {
        'dataset': 'CheXpert Plus Impression labels paired with original small CheXpert images',
        'image_source_repo': 'danjacobellis/chexpert',
        'image_source_revision': REVISION,
        'label_source_sha256': json.loads((ORIGINAL / 'manifest.json').read_text())['source_label_sha256'],
        'label_file': 'impression_fixed.json',
        'image_size': 'Original small CheXpert images resized to 320x320 at training time',
        'counts': counts,
        'sha256': hashes,
        'expert_valid_label_sha256': json.loads((ORIGINAL / 'expert_valid_manifest.json').read_text())['jsonl_sha256'],
        'test_result_known_before_experiment': True,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
