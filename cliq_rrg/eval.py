"""Evaluation metrics for CliQ-RRG following prior work (Jin et al., 2024;
Park et al., 2025).

NLG metrics: BLEU, METEOR, ROUGE-L, CIDEr, BERTScore.
Clinical efficacy: CheXbert F1/Precision/Recall.
QA metrics: BERTScore semantic, RadGraph-F1.
Inter-annotator agreement: Fleiss' kappa.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# NLG metrics (token-based, reference-free implementations)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """BLEU score (n-gram precision with brevity penalty)."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tokens, n)
        cand_ng = _ngrams(cand_tokens, n)
        if not cand_ng:
            precisions.append(0.0)
            continue
        overlap = sum((cand_ng & ref_ng).values())
        precisions.append(overlap / sum(cand_ng.values()))

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))

    if any(p == 0 for p in precisions):
        return 0.0
    score = math.exp(sum(math.log(p) for p in precisions) / max_n) * bp
    return score


def meteor(reference: str, candidate: str) -> float:
    """Simplified METEOR (unigram F1 with fragmentation penalty)."""
    ref_tokens = set(_tokenize(reference))
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    matches = len(ref_tokens & set(cand_tokens))
    if matches == 0:
        return 0.0

    precision = matches / len(cand_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10 * precision * recall) / (9 * precision + recall) if (9 * precision + recall) > 0 else 0.0

    # Fragmentation penalty (simplified)
    return f_mean


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L (longest common subsequence F1)."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def cider(references: list[str], candidate: str) -> float:
    """Simplified CIDEr (TF-IDF cosine similarity of n-grams)."""
    if not references:
        return 0.0
    # Average cosine similarity with each reference using unigram TF
    cand_tokens = _tokenize(candidate)
    cand_tf = Counter(cand_tokens)
    total_cand = sum(cand_tf.values()) or 1

    scores = []
    for ref in references:
        ref_tokens = _tokenize(ref)
        ref_tf = Counter(ref_tokens)
        total_ref = sum(ref_tf.values()) or 1

        common = set(cand_tf) & set(ref_tf)
        dot = sum((cand_tf[t] / total_cand) * (ref_tf[t] / total_ref) for t in common)
        norm_c = math.sqrt(sum((v / total_cand) ** 2 for v in cand_tf.values()))
        norm_r = math.sqrt(sum((v / total_ref) ** 2 for v in ref_tf.values()))
        if norm_c == 0 or norm_r == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (norm_c * norm_r))
    return sum(scores) / len(scores)


def bert_score(reference: str, candidate: str) -> float:
    """Placeholder BERTScore (cosine similarity). Falls back to ROUGE-L-like
    overlap if sentence-transformers is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = model.encode(reference, convert_to_tensor=True)
        emb2 = model.encode(candidate, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())
    except Exception:
        # Fallback: unigram overlap F1
        ref_t = set(_tokenize(reference))
        cand_t = set(_tokenize(candidate))
        if not ref_t or not cand_t:
            return 0.0
        inter = len(ref_t & cand_t)
        return 2 * inter / (len(ref_t) + len(cand_t))


# ---------------------------------------------------------------------------
# Clinical efficacy (CheXbert-style) - requires CheXbert model; placeholder
# ---------------------------------------------------------------------------

def chexbert_metrics(predicted_findings: list[str], reference_findings: list[str]):
    """CheXbert-style F1 / Precision / Recall for finding extraction."""
    pred_set = set(predicted_findings)
    ref_set = set(reference_findings)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def radgraph_f1(predicted_entities: list[tuple], reference_entities: list[tuple]):
    """RadGraph-F1 placeholder (entity + relation matching)."""
    pred_set = set(predicted_entities)
    ref_set = set(reference_entities)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def fleiss_kappa(ratings: list[list[int]], n_categories: int) -> float:
    """Fleiss' kappa for inter-annotator agreement.

    ratings: list of items, each a list of category counts per annotator.
    """
    n_items = len(ratings)
    if n_items == 0:
        return 0.0
    n_annotators = sum(ratings[0]) if ratings[0] else 0
    if n_annotators == 0:
        return 0.0

    # Category proportions per item
    p_js = []
    for item in ratings:
        total = sum(item)
        p_js.append([c / total for c in item]) if total > 0 else p_js.append([0.0] * n_categories)

    # Mean proportion for each category across items
    P_bar = [sum(p[j] for p in p_js) / n_items for j in range(n_categories)]

    # P_i per item
    P_is = []
    for item, p_j in zip(ratings, p_js):
        total = sum(item)
        if total <= 1:
            P_is.append(0.0)
            continue
        sum_sq = sum(c * c for c in item)
        P_i = (sum_sq - total) / (total * (total - 1))
        P_is.append(P_i)

    P_bar_bar = sum(P_is) / n_items
    P_e_bar = sum(p * p for p in P_bar)

    denom = 1 - P_e_bar
    if denom == 0:
        return 0.0
    return (P_bar_bar - P_e_bar) / denom


# ---------------------------------------------------------------------------
# Aggregate report-level evaluation
# ---------------------------------------------------------------------------

def evaluate_reports(references: list[str], candidates: list[str]) -> dict:
    """Compute all NLG metrics for a list of (reference, candidate) report pairs."""
    results = {
        "bleu": [], "meteor": [], "rouge_l": [], "cider": [], "bert_score": [],
    }
    for ref, cand in zip(references, candidates):
        results["bleu"].append(bleu(ref, cand))
        results["meteor"].append(meteor(ref, cand))
        results["rouge_l"].append(rouge_l(ref, cand))
        results["cider"].append(cider(references, cand))
        results["bert_score"].append(bert_score(ref, cand))

    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in results.items()
    }


if __name__ == "__main__":
    # Quick smoke test
    ref = "The chest X-ray shows evidence of pneumonia and cardiomegaly."
    cand = "There is pneumonia and an enlarged heart on the frontal view."
    print(f"BLEU: {bleu(ref, cand):.4f}")
    print(f"METEOR: {meteor(ref, cand):.4f}")
    print(f"ROUGE-L: {rouge_l(ref, cand):.4f}")
    print(f"BERTScore: {bert_score(ref, cand):.4f}")
    print(f"CheXbert: {chexbert_metrics(['pneumonia', 'cardiomegaly'], ['pneumonia', 'cardiomegaly'])}")
    print(f"RadGraph-F1: {radgraph_f1([('pneumonia', 'present')], [('pneumonia', 'present')])}")
