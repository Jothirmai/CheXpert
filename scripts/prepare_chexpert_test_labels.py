"""Freeze the public 500-study CheXpert expert test labels without using them for training."""

import csv
import hashlib
import io
import json
from pathlib import Path

import requests


REVISION = 'a03eb3438ca55c28eb4379dd0dd09e3b38c0c414'
URL = f'https://raw.githubusercontent.com/rajpurkarlab/cheXpert-test-set-labels/{REVISION}/groundtruth.csv'
OUT = Path('data/chexpert_test_500')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'groundtruth.csv'
    if target.exists():
        content = target.read_bytes()
    else:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        content = response.content
    rows = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    if len(rows) != 500 or len({row['Study'] for row in rows}) != 500:
        raise RuntimeError('Expected 500 distinct expert-labeled studies')
    if any(not row['Study'].startswith('CheXpert-v1.0/test/patient') for row in rows):
        raise RuntimeError('Unexpected test study path')
    findings = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    counts = {}
    for finding in findings:
        values = [row[finding] for row in rows]
        if not set(values) <= {'0', '1'}:
            raise RuntimeError(f'Nonbinary expert label for {finding}')
        counts[finding] = {'positive': values.count('1'), 'negative': values.count('0')}
    target.write_bytes(content)
    manifest = {
        'source': URL,
        'source_revision': REVISION,
        'sha256': hashlib.sha256(content).hexdigest(),
        'studies': len(rows),
        'findings': findings,
        'counts': counts,
        'images_available': False,
        'use': 'evaluation only; never train or select models with these labels',
        'image_source': 'Stanford AIMI CheXlocalize download; requires separate authorized access',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
