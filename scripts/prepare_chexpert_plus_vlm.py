"""Freeze patient-separated CheXpert Plus report pairs for VLM fine-tuning.

Report text is read from the authorized Redivis release. Images come from the
locally pinned original CheXpert JPEG snapshot. No report text is logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import redivis

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/chexpert_plus_cnn_impression_320_v1'
OUT = ROOT / 'data/chexpert_plus_vlm_v1'
SOURCE_TABLE = 'aimi.chexpert_plus:5yyj:v1_0.df_chexpert_plus_240401:bavj'
TARGETS = {'train': 2048, 'validation': 96, 'test': 96}
QUERY = ("SELECT path_to_image, section_findings, section_impression "
         "FROM df_chexpert_plus_240401:bavj "
         "WHERE split = 'train' AND LOWER(frontal_lateral) = 'frontal' "
         "AND section_findings IS NOT NULL AND LENGTH(TRIM(section_findings)) >= 20 "
         "AND section_impression IS NOT NULL AND LENGTH(TRIM(section_impression)) >= 10 "
         "ORDER BY MD5(path_to_image) LIMIT 12000")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_text(value):
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', ' ', str(value)).strip()


def local_rows(name):
    path = SOURCE / f'{name}.jsonl'
    rows = [json.loads(line) for line in path.open()]
    return {row['source_path']: row for row in rows}


def main():
    if not os.environ.get('REDIVIS_API_TOKEN'):
        raise RuntimeError('REDIVIS_API_TOKEN is required to read CheXpert Plus reports')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Frozen CheXpert Plus VLM dataset already exists')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for name in ('fit', 'development'):
        if sha(SOURCE / f'{name}.jsonl') != source_manifest['sha256'][name]:
            raise RuntimeError(f'CNN source {name} manifest changed')
    fit = local_rows('fit')
    development = local_rows('development')
    frame = redivis.query(QUERY, default_dataset='aimi.chexpert_plus:5yyj:v1_0').to_pandas_dataframe(progress=False)
    groups = {'train': [], 'development': []}
    for record in frame.itertuples(index=False):
        path, findings, impression = record
        local = fit.get(path)
        group = 'train'
        if local is None:
            local = development.get(path)
            group = 'development'
        if local is None or not Path(local['image_path']).is_file():
            continue
        findings, impression = clean_text(findings), clean_text(impression)
        if len(findings) < 20 or len(impression) < 10:
            continue
        study = path.rsplit('/', 1)[0]
        groups[group].append((study, local, findings, impression))
    # Stable query order chooses one frontal view per study and at most one
    # study per patient. This avoids repeating identical report targets.
    def choose(candidates, wanted, excluded):
        selected, patients, studies = [], set(), set()
        for study, local, findings, impression in candidates:
            patient = local['patient_id']
            if patient in patients or patient in excluded or study in studies:
                continue
            patients.add(patient)
            studies.add(study)
            image = Path(local['image_path'])
            selected.append({
                'dataset': 'chexpert_plus', 'study_id': f'CHEXPERT:{study}',
                'patient_id': patient, 'source_path': local['source_path'],
                'image_path': str(image), 'image_paths': [str(image)],
                'image_sha256': sha(image),
                'clinical_indication': None, 'comparison_study': None,
                'technique': None, 'mesh_terms': [],
                'target': {'sections': [
                    {'heading': 'Findings', 'text': findings},
                    {'heading': 'Impression', 'text': impression},
                ]},
            })
            if len(selected) == wanted:
                break
        if len(selected) != wanted:
            raise RuntimeError(f'Only {len(selected)}/{wanted} eligible distinct-patient report pairs')
        return selected, patients
    train, train_patients = choose(groups['train'], TARGETS['train'], set())
    validation, val_patients = choose(groups['development'], TARGETS['validation'], set())
    test, test_patients = choose(groups['development'], TARGETS['test'], val_patients)
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise RuntimeError('Patient overlap across VLM splits')
    hashes = {}
    for name, rows in [('train', train), ('validation', validation), ('test', test)]:
        path = OUT / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        hashes[name] = sha(path)
    manifest = {
        'dataset': 'CheXpert Plus radiology reports paired with original CheXpert small frontal JPEGs',
        'source_table': SOURCE_TABLE,
        'source_query_sha256': hashlib.sha256(QUERY.encode()).hexdigest(),
        'source_query': QUERY,
        'selection': 'MD5(path_to_image) query order; one frontal image and one study per patient; required nonempty Findings and Impression',
        'source_image_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'cnn_fit_patients_used_for_vlm_train': True,
        'cnn_development_patients_used_for_vlm_validation_and_test': True,
        'test_not_used_for_vlm_training_or_selection': True,
        'image_role': 'original small CheXpert JPEG, not 224-pixel CNN cache',
        'report_role': 'CheXpert Plus deidentified Findings and Impression',
        'counts': {name: len(rows) for name, rows in [('train', train), ('validation', validation), ('test', test)]},
        'patient_counts': {name: len({row['patient_id'] for row in rows}) for name, rows in [('train', train), ('validation', validation), ('test', test)]},
        'sha256': hashes,
        'limitations': 'Local VLM split differs from published CheXpert Plus report-generation benchmarks; cross-paper scores are contextual until evaluated on the same split and scorer.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'phase': 'complete', 'counts': manifest['counts'],
                      'patient_counts': manifest['patient_counts'], 'source_rows': len(frame),
                      'eligible_local_rows': {name: len(rows) for name, rows in groups.items()}}), flush=True)


if __name__ == '__main__':
    main()
