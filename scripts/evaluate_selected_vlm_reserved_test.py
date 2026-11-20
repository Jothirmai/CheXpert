"""Run the selected draft VLM once on the frozen CheXpert Plus test split.

The test split is used for reporting only. Results never change model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chest_xpert.label_mapping import LabelMapping
from chest_xpert.report_scoring_v2 import score as clinical_score, summarize
from chest_xpert.verification import LLMModelClient
from scripts.score_vlm_pair_standard import clinical_text, nlg_metrics, strict_prediction
from scripts.train_report_adapter import enrich

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/chexpert_plus_vlm_v1'
RUN = ROOT / 'checkpoints/report_lora_chexpert_plus_v1'
ROWS = DATA / 'test.jsonl'
PARTIAL = RUN / 'reserved_test_generations.jsonl'
RESULT = RUN / 'reserved_test_results.json'
PROGRESS = RUN / 'reserved_test_progress.json'


def write_progress(path, phase, completed, total):
    path.write_text(json.dumps({'phase': phase, 'completed': completed,
                                    'total': total, 'percent': round(completed / total * 100, 2)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--followup-fp16', action='store_true')
    args = parser.parse_args()
    partial = RUN / 'reserved_test_fp16_generations.jsonl' if args.followup_fp16 else PARTIAL
    result_path = RUN / 'reserved_test_fp16_followup.json' if args.followup_fp16 else RESULT
    progress_path = RUN / 'reserved_test_fp16_progress.json' if args.followup_fp16 else PROGRESS
    if result_path.exists():
        raise FileExistsError('Frozen reserved-test result already exists')
    if args.followup_fp16 and not (RUN / 'reserved_test_results_rescored.json').exists():
        raise RuntimeError('FP16 follow-up requires the completed original 4-bit test')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    digest = hashlib.sha256(ROWS.read_bytes()).hexdigest()
    if digest != manifest['sha256']['test']:
        raise RuntimeError('Frozen test manifest changed')
    selection = json.loads((ROOT / 'checkpoints/active_models.json').read_text())['report_adapter']
    if selection['path'] != str(RUN.relative_to(ROOT)):
        raise RuntimeError('Selected adapter changed')
    adapter_hash = hashlib.sha256((RUN / 'adapter_model.safetensors').read_bytes()).hexdigest()
    if adapter_hash != selection['adapter_sha256']:
        raise RuntimeError('Selected adapter weights changed')
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    selected = json.loads((RUN / 'selected_studies.json').read_text())
    if {r['study_id'] for r in rows} & (set(selected['train']) | set(selected['validation'])):
        raise RuntimeError('Test study overlaps training or validation')
    completed = {r['study_id']: r for r in map(json.loads, partial.read_text().splitlines())} if partial.exists() else {}
    write_progress(progress_path, 'preparing_evidence', len(completed), len(rows))
    remaining = [r for r in rows if r['study_id'] not in completed]
    if remaining:
        remaining = enrich(remaining, ROOT / 'data/report_training_v3/evidence', 'cuda:0', 'selected_ensemble')
        client = LLMModelClient(load_in_4bit=False if args.followup_fp16 else None)
        if not client.is_configured() or not client._lora_loaded:
            raise RuntimeError(f'Selected VLM adapter could not load: {client._load_error}')
        write_progress(progress_path, 'generating', len(completed), len(rows))
        for row in remaining:
            mappings = [LabelMapping(name, True, name, value)
                        for name, value in row['cnn_scores'].items()]
            patient_info = {
                'report_sections': [s['heading'] for s in row['target']['sections']],
                'clinical_indication': row.get('clinical_indication') or 'NOT PROVIDED',
                'comparison_study': row.get('comparison_study') or 'NOT PROVIDED',
                'technique': row.get('technique') or 'NOT PROVIDED',
            }
            generation = client.generate_v7_report(
                mappings, 'Selected DenseNet201 ensemble; scores are uncalibrated.',
                patient_info, row['image_paths'], row['contact_sheet_path'], self_correct=False)
            output = {
                'study_id': row['study_id'], 'reference': row['target'],
                'prediction': generation.v7_report or '',
                'generation_error': generation.error,
            }
            with partial.open('a') as stream:
                stream.write(json.dumps(output, ensure_ascii=False) + '\n')
            completed[row['study_id']] = output
            write_progress(progress_path, 'generating', len(completed), len(rows))
            print(f"TEST {len(completed)}/{len(rows)} {row['study_id']}", flush=True)
    ordered = [completed[r['study_id']] for r in rows]
    references = [clinical_text(r['reference']['sections']) for r in ordered]
    predictions = [strict_prediction(r['prediction'],
                     [s['heading'] for s in r['reference']['sections']]) for r in ordered]
    clinical = [clinical_score(r['prediction'], {'target': r['reference']}) for r in ordered]
    write_progress(progress_path, 'scoring', len(ordered), len(ordered))
    result = {
        'split': 'frozen patient-disjoint CheXpert Plus local reserved test',
        'test_manifest_sha256': digest,
        'adapter_sha256': adapter_hash,
        'cases': len(ordered),
        'schema_valid': sum(p is not None for p in predictions),
        'metrics': nlg_metrics(references, predictions, bert=True),
        'clinical_audit': summarize(clinical),
        'selection_changed_by_test': False,
        'runtime_precision': 'fp16' if args.followup_fp16 else '4bit',
        'evaluation_status': 'exploratory follow-up after original test inspection' if args.followup_fp16 else 'first frozen reserved-test pass',
        'limitations': 'Local test split and lexical clinical checks; no expert image-based accuracy adjudication and no matched paper benchmark.',
    }
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    write_progress(progress_path, 'complete', len(ordered), len(ordered))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
