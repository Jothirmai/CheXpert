"""Score saved base/LoRA report pairs with pinned, named metric libraries.

Scores are local split results. They must not be ranked against paper scores
obtained on different patients or with different report section targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def clinical_text(sections):
    return ' '.join(str(section['text']).strip() for section in sections
                    if section['heading'] in ('Findings', 'Impression')).strip()


def strict_prediction(raw, headings):
    try:
        normalized = raw.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', normalized, flags=re.IGNORECASE)
        if fenced:
            normalized = fenced.group(1)
        data = json.loads(normalized)
        if set(data) == {'sections'}:
            sections = data['sections']
        elif set(data) == {'report_sections'}:
            sections = data['report_sections']
        elif set(data) == {heading.lower() for heading in headings}:
            sections = [{'heading': heading, 'text': data[heading.lower()]}
                        for heading in headings]
        else:
            return None
        if not isinstance(sections, list) or [s['heading'] for s in sections] != headings:
            return None
        if any(not isinstance(s['text'], str) for s in sections):
            return None
        return clinical_text(sections)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def nlg_metrics(references, candidates, bert=False):
    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize({str(i): [{'caption': ref}] for i, ref in enumerate(references)})
    preds = tokenizer.tokenize({str(i): [{'caption': text or ''}] for i, text in enumerate(candidates)})
    bleu = Bleu(4).compute_score(refs, preds)[0]
    meteor = Meteor()
    try:
        meteor_value = float(meteor.compute_score(refs, preds)[0])
    finally:
        # The installed pycocoevalcap Meteor wrapper has no close() method.
        # It owns a Java subprocess, so close its pipes and reap it explicitly.
        process = meteor.meteor_p
        process.stdin.close()
        process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
    result = {
        'BLEU_1': float(bleu[0]), 'BLEU_4': float(bleu[3]),
        'METEOR': meteor_value,
        'ROUGE_L': float(Rouge().compute_score(refs, preds)[0]),
        'CIDEr': float(Cider().compute_score(refs, preds)[0]),
    }
    if bert:
        from bert_score import score
        valid = [i for i, text in enumerate(candidates) if text]
        if valid:
            _, _, f1 = score([candidates[i] for i in valid], [references[i] for i in valid],
                             model_type='roberta-large', lang='en', verbose=False,
                             batch_size=8, rescale_with_baseline=False)
            result['BERTScore_F1_roberta_large'] = float(f1.mean()) * len(valid) / len(candidates)
        else:
            result['BERTScore_F1_roberta_large'] = 0.0
    else:
        result['BERTScore_F1_roberta_large'] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--bert-score', action='store_true')
    args = parser.parse_args()
    folder = args.experiment
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    cases = evaluation['comparisons']
    references = []
    candidates = {'base': [], 'adapter': []}
    details = []
    for case in cases:
        sections = case['reference']['sections']
        headings = [s['heading'] for s in sections]
        reference = clinical_text(sections)
        if not reference:
            raise RuntimeError(f"Missing reference clinical text: {case['study_id']}")
        references.append(reference)
        detail = {'study_id': case['study_id'], 'reference': reference}
        for mode in candidates:
            prediction = strict_prediction(case[f'{mode}_output'], headings)
            detail[mode] = {'schema_valid': prediction is not None,
                            'clinical_text': prediction}
            candidates[mode].append(prediction)
        details.append(detail)
    metrics = {}
    for mode, texts in candidates.items():
        metrics[mode] = {'cases': len(cases),
                         'strict_json_valid': sum(text is not None for text in texts),
                         **nlg_metrics(references, texts, args.bert_score)}
    result = {
        'metrics': metrics, 'cases': details,
        'reference_scope': 'Findings plus Impression only',
        'candidate_scope': 'Requested clinical headings, allowing one Markdown JSON fence and lossless JSON key normalization; invalid outputs scored as empty text',
        'scorers': {'BLEU_METEOR_ROUGE_L_CIDEr': 'pycocoevalcap with PTBTokenizer',
                    'BERTScore': 'bert_score roberta-large F1 without baseline rescaling' if args.bert_score else 'not run'},
        'input_evidence_sha256': hashlib.sha256((folder / 'input_evidence.json').read_bytes()).hexdigest(),
        'limitations': 'Saved local validation subset only. No claim of matched published benchmark split or independently adjudicated clinical correctness.',
    }
    (folder / 'standard_validation_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
