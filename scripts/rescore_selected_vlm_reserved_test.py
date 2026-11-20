"""Rescore frozen saved test generations after lossless JSON parser correction.

The original result is preserved; no model output is regenerated or edited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_scoring_v2 import score as clinical_score, summarize
from scripts.score_vlm_pair_standard import clinical_text, nlg_metrics, strict_prediction

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'checkpoints/report_lora_chexpert_plus_v1'
SOURCE = RUN / 'reserved_test_results.json'
DEST = RUN / 'reserved_test_results_rescored.json'


def main():
    if DEST.exists():
        raise FileExistsError('Reserved-test rescore already exists')
    original = json.loads(SOURCE.read_text())
    cases = [json.loads(line) for line in (RUN / 'reserved_test_generations.jsonl').read_text().splitlines()]
    if len(cases) != original['cases'] or len({c['study_id'] for c in cases}) != len(cases):
        raise RuntimeError('Saved generation cohort mismatch')
    references = [clinical_text(c['reference']['sections']) for c in cases]
    predictions = [strict_prediction(c['prediction'],
                   [s['heading'] for s in c['reference']['sections']]) for c in cases]
    clinical = [clinical_score(c['prediction'], {'target': c['reference']}) for c in cases]
    result = {**original,
              'schema_valid': sum(p is not None for p in predictions),
              'metrics': nlg_metrics(references, predictions, bert=True),
              'clinical_audit': summarize(clinical),
              'rescore_reason': 'Lossless acceptance of report_sections and lowercase heading-key JSON; raw model generations unchanged.',
              'original_result': str(SOURCE.relative_to(ROOT))}
    DEST.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('cases', 'schema_valid', 'metrics', 'clinical_audit')}, indent=2))


if __name__ == '__main__':
    main()
