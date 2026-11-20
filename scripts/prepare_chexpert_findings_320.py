"""Materialize 320-pixel CheXpert images for the frozen Findings-section split."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


REVISION = 'ca5a840e438b5d2d0fa07f560782f196b38a84aa'
SOURCE = Path.home() / '.cache/huggingface/hub/datasets--danjacobellis--chexpert/snapshots' / REVISION / 'data'
FINDINGS = Path('data/chexpert_plus_cnn_findings_subset_v1')
IMPRESSION = Path('data/chexpert_plus_cnn_impression_v1')
OUT = Path('data/chexpert_plus_cnn_findings_320_v1')


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def materialize(files, wanted):
    found = set()
    for index, source in enumerate(files, 1):
        parquet = pq.ParquetFile(source)
        for group in range(parquet.num_row_groups):
            batch = parquet.read_row_group(group, columns=['Path', 'image'])
            for row in batch.to_pylist():
                relative = row['Path'].removeprefix('CheXpert-v1.0-small/')
                if relative not in wanted:
                    continue
                target = OUT / 'images' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(row['image']['bytes'])
                found.add(relative)
        print('SOURCE', index, '/', len(files), 'matched', len(found), flush=True)
    if found != wanted:
        raise RuntimeError(f'Missing {len(wanted - found)} required images from mirror')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Frozen 320-pixel cache already exists')
    splits = {
        'fit': load_rows(FINDINGS / 'fit.jsonl'),
        'development': load_rows(FINDINGS / 'development.jsonl'),
        'valid': load_rows(IMPRESSION / 'valid.jsonl'),
        'expert_valid': load_rows(IMPRESSION / 'expert_valid.jsonl'),
    }
    train_wanted = {row['source_path'] for name in ('fit', 'development') for row in splits[name]}
    valid_wanted = {row['source_path'] for row in splits['valid']}
    train_files = [SOURCE / f'train-{index:05d}-of-00023.parquet' for index in range(23)]
    valid_files = [SOURCE / 'validation-00000-of-00001.parquet']
    if not all(path.is_file() for path in train_files + valid_files):
        raise FileNotFoundError('The pinned CheXpert parquet snapshot is incomplete')
    materialize(train_files, train_wanted)
    materialize(valid_files, valid_wanted)
    hashes = {}
    counts = {}
    for name, rows in splits.items():
        for row in rows:
            row['image_path'] = str((OUT / 'images' / row['source_path']).resolve())
        path = OUT / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[name] = {'images': len(rows), 'patients': len({row['patient_id'] for row in rows})}
    fit_patients = {row['patient_id'] for row in splits['fit']}
    dev_patients = {row['patient_id'] for row in splits['development']}
    valid_patients = {row['patient_id'] for row in splits['valid']}
    if fit_patients & dev_patients or fit_patients & valid_patients or dev_patients & valid_patients:
        raise RuntimeError('Patient overlap among frozen splits')
    result = {
        'dataset': 'CheXpert-v1.0-small image mirror, paired with CheXpert Plus Findings-section labels',
        'image_source_repo': 'danjacobellis/chexpert',
        'image_source_revision': REVISION,
        'label_source_sha256': json.loads((FINDINGS / 'manifest.json').read_text())['source_label_sha256'],
        'label_file': 'findings_fixed.json',
        'image_size': '320-pixel original small CheXpert images; resized to 320x320 at training time',
        'counts': counts,
        'sha256': hashes,
        'expert_valid_label_sha256': json.loads((IMPRESSION / 'expert_valid_manifest.json').read_text())['jsonl_sha256'],
        'test_used': False,
    }
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
