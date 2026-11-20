"""Audit saved validation generations against reference assertions.

This lexical comparison is a review aid, not image-based clinical adjudication.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_scoring_v2 import VERSION, score, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.experiment / 'evaluation.json').read_text())['comparisons']
    details = []
    for case in cases:
        reference = {'target': case['reference']}
        details.append({
            'study_id': case['study_id'],
            'base': score(case['base_output'], reference),
            'adapter': score(case['adapter_output'], reference),
        })
    result = {
        'scorer_version': VERSION,
        'scope': 'Saved validation reports only',
        'summary': {model: summarize([case[model] for case in details])
                    for model in ('base', 'adapter')},
        'clinical_validation_passed': False,
        'limitations': 'Lexical assertion matching is incomplete and cannot establish image-grounded clinical accuracy. Every generated report requires human review.',
        'cases': details,
    }
    path = args.experiment / 'clinical_validation_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(path), 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
