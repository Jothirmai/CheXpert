"""Freeze the expert-labeled CheXpert validation rows matching our 202 frontal images."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path('data/chexpert_plus_cnn_impression_v1')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
REPO = 'danjacobellis/chexpert'
REVISION = 'ca5a840e438b5d2d0fa07f560782f196b38a84aa'
FILENAME = 'data/validation-00000-of-00001.parquet'


def main():
    out = ROOT / 'expert_valid.jsonl'
    manifest_path = ROOT / 'expert_valid_manifest.json'
    if out.exists() or manifest_path.exists():
        raise RuntimeError('Expert validation manifest already frozen')
    parquet = Path(hf_hub_download(REPO, FILENAME, repo_type='dataset', revision=REVISION))
    table = pq.read_table(parquet, columns=['Path', *FINDINGS]).to_pylist()
    by_path = {row['Path'].replace('CheXpert-v1.0-small/', ''): row for row in table}
    if len(by_path) != len(table):
        raise RuntimeError('Duplicate source path')
    valid = [json.loads(line) for line in (ROOT / 'valid.jsonl').read_text().splitlines()]
    rows = []
    for row in valid:
        expert = by_path.get(row['source_path'])
        if expert is None:
            raise RuntimeError(f"No expert label for {row['source_path']}")
        labels = {}
        for finding in FINDINGS:
            code = int(expert[finding])
            labels[finding] = {0: None, 1: -1, 2: 0, 3: 1}[code]
        rows.append({'source_path': row['source_path'], 'image_path': row['image_path'], 'patient_id': row['patient_id'], 'labels': labels})
    out.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    counts = {finding: {'positive': sum(row['labels'][finding] == 1 for row in rows), 'negative': sum(row['labels'][finding] == 0 for row in rows), 'uncertain': sum(row['labels'][finding] == -1 for row in rows), 'unlabeled': sum(row['labels'][finding] is None for row in rows)} for finding in FINDINGS}
    manifest = {'source_repo': REPO, 'source_revision': REVISION, 'source_file': FILENAME, 'source_parquet_sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(), 'matched_to': 'CheXpert Plus official valid frontal images', 'images': len(rows), 'patients': len({row['patient_id'] for row in rows}), 'findings': FINDINGS, 'counts': counts, 'jsonl_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'test_used': False}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'images': len(rows), 'patients': manifest['patients'], 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
