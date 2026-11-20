"""Freeze the subset with a real Findings section and its CheXbert labels."""
import hashlib
import json
import os
from pathlib import Path

import redivis

SOURCE = Path('data/chexpert_plus_cnn_impression_v1')
TARGET = Path('data/chexpert_plus_cnn_findings_subset_v1')
LABEL_FILE = 'findings_fixed.json'


def main():
    if not os.environ.get('REDIVIS_API_TOKEN'):
        raise RuntimeError('REDIVIS_API_TOKEN missing')
    TARGET.mkdir(parents=True, exist_ok=True)
    if (TARGET / 'manifest.json').exists():
        raise RuntimeError('Findings subset already frozen')
    dataset = redivis.user('aimi').dataset('chexpert_plus:5yyj:v1_0')
    sql = "SELECT path_to_image FROM df_chexpert_plus_240401:bavj WHERE section_findings IS NOT NULL AND LENGTH(TRIM(section_findings)) >= 2 AND LOWER(frontal_lateral) = 'frontal'"
    available = set(redivis.query(sql, default_dataset='aimi.chexpert_plus:5yyj:v1_0').to_pandas_dataframe(progress=False)['path_to_image'].astype(str))
    file = dataset.table('chexpert_labels:pmec').file(LABEL_FILE)
    raw = file.read()
    if len(raw) != file.size:
        raise RuntimeError('Incomplete findings label file')
    labels = json.loads((SOURCE / 'manifest.json').read_text())['labels']
    by_path = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if row['path_to_image'] in available:
            by_path[row['path_to_image']] = [row.get(name) for name in labels]
    if set(by_path) != available:
        raise RuntimeError(f'{len(available - set(by_path))} real findings sections have no label record')
    counts = {}
    hashes = {}
    for split in ('fit', 'development', 'valid'):
        rows = []
        for line in (SOURCE / f'{split}.jsonl').open():
            row = json.loads(line)
            if row['source_path'] not in by_path:
                continue
            row['labels'] = by_path[row['source_path']]
            rows.append(row)
        path = TARGET / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
        counts[split] = {'images': len(rows), 'patients': len({row['patient_id'] for row in rows})}
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    fit_patients = {json.loads(line)['patient_id'] for line in (TARGET / 'fit.jsonl').open()}
    dev_patients = {json.loads(line)['patient_id'] for line in (TARGET / 'development.jsonl').open()}
    valid_patients = {json.loads(line)['patient_id'] for line in (TARGET / 'valid.jsonl').open()}
    assert fit_patients.isdisjoint(dev_patients) and fit_patients.isdisjoint(valid_patients) and dev_patients.isdisjoint(valid_patients)
    manifest = {'dataset': 'CheXpert Plus authorized Redivis release', 'label_file': LABEL_FILE, 'source_label_sha256': hashlib.sha256(raw).hexdigest(), 'source_split_sha256': json.loads((SOURCE / 'development_split.json').read_text())['sha256'], 'filter': 'nonempty section_findings and frontal image', 'labels': labels, 'counts': counts, 'sha256': hashes, 'official_valid_untouched': True, 'test_used': False}
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
