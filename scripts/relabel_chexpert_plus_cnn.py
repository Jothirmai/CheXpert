"""Replace misaligned full-report labels with CheXpert Plus impression labels."""
import hashlib
import json
import os
from pathlib import Path

import redivis

SOURCE = Path('data/chexpert_plus_cnn_v1')
TARGET = Path('data/chexpert_plus_cnn_impression_v1')
LABEL_FILE = 'impression_fixed.json'


def main():
    if not os.environ.get('REDIVIS_API_TOKEN'):
        raise RuntimeError('REDIVIS_API_TOKEN missing')
    TARGET.mkdir(parents=True, exist_ok=True)
    if (TARGET / 'manifest.json').exists():
        raise RuntimeError('Frozen target manifest exists')
    dataset = redivis.user('aimi').dataset('chexpert_plus:5yyj:v1_0')
    file = dataset.table('chexpert_labels:pmec').file(LABEL_FILE)
    raw = file.read()
    if len(raw) != file.size:
        raise RuntimeError('Incomplete label file')
    labels = json.loads((SOURCE / 'manifest.json').read_text())['labels']
    by_path = {}
    for line in raw.splitlines():
        row = json.loads(line)
        by_path[row['path_to_image']] = [row.get(label) for label in labels]
    if len(by_path) != len(raw.splitlines()):
        raise RuntimeError('Duplicate image paths in source label file')
    counts = {}
    hashes = {}
    for split in ('train', 'valid'):
        old = [json.loads(line) for line in (SOURCE / f'{split}.jsonl').read_text().splitlines()]
        missing = [row['source_path'] for row in old if row['source_path'] not in by_path]
        if missing:
            raise RuntimeError(f'{len(missing)} image paths absent from {LABEL_FILE}')
        for row in old:
            row['labels'] = by_path[row['source_path']]
        path = TARGET / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in old))
        counts[split] = {'images': len(old), 'patients': len({row['patient_id'] for row in old})}
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'dataset': 'CheXpert Plus authorized Redivis release',
        'labels': labels,
        'label_file': LABEL_FILE,
        'source_label_sha256': hashlib.sha256(raw).hexdigest(),
        'source_image_manifest_sha256': hashlib.sha256((SOURCE / 'manifest.json').read_bytes()).hexdigest(),
        'frontal_only': True,
        'split': 'official train/valid patient-disjoint',
        'counts': counts,
        'manifest_sha256': hashes,
        'test_used': False,
        'reason': 'The full-report label file disagreed with corresponding report impressions on audited samples; the impression-specific labels matched those impressions.',
    }
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
