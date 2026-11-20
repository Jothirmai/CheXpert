"""Extract lateral views for the frozen patient-disjoint Impression split."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


REVISION = 'ca5a840e438b5d2d0fa07f560782f196b38a84aa'
SOURCE = Path.home() / '.cache/huggingface/hub/datasets--danjacobellis--chexpert/snapshots' / REVISION / 'data'
FRONTAL = Path('data/chexpert_plus_cnn_impression_320_v1')
OUT = Path('data/chexpert_plus_cnn_lateral_320_v1')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Lateral cache is already frozen')
    frontal_manifest = json.loads((FRONTAL / 'manifest.json').read_text())
    by_split = {}
    conflicts = {}
    all_studies = {}
    for split in ('fit', 'development', 'valid'):
        source = FRONTAL / f'{split}.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest() != frontal_manifest['sha256'][split]:
            raise RuntimeError(f'Frozen frontal {split} split changed')
        groups = defaultdict(list)
        for line in source.read_text().splitlines():
            row = json.loads(line)
            groups[row['source_path'].rsplit('/', 1)[0]].append(row)
        conflicting = {study for study, rows in groups.items()
                       if any(row['labels'] != rows[0]['labels'] for row in rows[1:])}
        by_split[split] = {study: rows[0] for study, rows in groups.items() if study not in conflicting}
        conflicts[split] = len(conflicting)
        for study in by_split[split]:
            if study in all_studies:
                raise RuntimeError(f'Study crosses splits: {study}')
            all_studies[study] = split
    files = [SOURCE / f'train-{i:05d}-of-00023.parquet' for i in range(23)]
    files.append(SOURCE / 'validation-00000-of-00001.parquet')
    if not all(path.is_file() for path in files):
        raise FileNotFoundError('Pinned CheXpert parquet snapshot is incomplete')
    output_rows = {split: [] for split in by_split}
    for number, file in enumerate(files, 1):
        parquet = pq.ParquetFile(file)
        for group in range(parquet.num_row_groups):
            batch = parquet.read_row_group(group, columns=['Path', 'image'])
            for item in batch.to_pylist():
                relative = item['Path'].removeprefix('CheXpert-v1.0-small/')
                if 'lateral' not in relative.rsplit('/', 1)[-1].lower():
                    continue
                study = relative.rsplit('/', 1)[0]
                split = all_studies.get(study)
                if split is None:
                    continue
                target = OUT / 'images' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(item['image']['bytes'])
                reference = by_split[split][study]
                output_rows[split].append({
                    'source_path': relative,
                    'image_path': str(target.resolve()),
                    'patient_id': reference['patient_id'],
                    'labels': reference['labels'],
                    'paired_frontal_source_path': reference['source_path'],
                })
        print('SOURCE', number, '/', len(files),
              {split: len(rows) for split, rows in output_rows.items()}, flush=True)
    hashes = {}
    counts = {}
    for split, rows in output_rows.items():
        rows.sort(key=lambda row: row['source_path'])
        if not rows:
            raise RuntimeError(f'No lateral images found for {split}')
        path = OUT / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = {
            'images': len(rows),
            'studies': len({row['source_path'].rsplit('/', 1)[0] for row in rows}),
            'patients': len({row['patient_id'] for row in rows}),
        }
    patients = {split: {row['patient_id'] for row in rows} for split, rows in output_rows.items()}
    if any(patients[a] & patients[b] for a, b in (('fit', 'development'), ('fit', 'valid'), ('development', 'valid'))):
        raise RuntimeError('Patient overlap in lateral splits')
    result = {
        'dataset': 'CheXpert-v1.0-small lateral views with paired CheXpert Plus Impression study labels',
        'image_source_repo': 'danjacobellis/chexpert',
        'image_source_revision': REVISION,
        'label_source_sha256': frontal_manifest['label_source_sha256'],
        'label_file': frontal_manifest['label_file'],
        'counts': counts,
        'excluded_studies_with_discordant_frontal_labels': conflicts,
        'sha256': hashes,
        'patient_disjoint': True,
        'test_result_known_before_experiment': True,
        'limitation': 'Study report labels are weak supervision for individual lateral images.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
