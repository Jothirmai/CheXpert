"""Medical RAG: Retrieval-Augmented Generation for radiology reports.

Builds a disease-similarity index from IU X-Ray training reports. At
inference time, retrieves k most similar reports (by shared disease labels)
and prepends them as clinical evidence context to the LLM prompt, improving
report grounding and factual accuracy.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliq_rrg.dataset import IU_FINDINGS


class MedicalRAGIndex:
    """Disease-based retrieval index over IU X-Ray reports.

    Index maps each disease -> list of (study_id, report_text, findings).
    Retrieval: given a set of predicted diseases, find reports that share
    the most diseases, return top-k as clinical evidence context.
    """

    def __init__(self):
        # disease -> list of (study_id, report_text, findings_set)
        self.disease_index: dict[str, list[tuple[str, str, set[str]]]] = {
            d: [] for d in IU_FINDINGS
        }
        self.all_reports: list[dict] = []
        self._built = False

    def add_report(self, study_id: str, report_text: str, findings: list[str]):
        findings_set = set(findings)
        self.all_reports.append({
            "study_id": study_id,
            "report_text": report_text,
            "findings": findings,
        })
        for d in findings:
            if d in self.disease_index:
                self.disease_index[d].append((study_id, report_text, findings_set))

    def build(self, samples: list):
        """Build index from a list of IUSample objects."""
        for s in samples:
            findings = [f for f in IU_FINDINGS if f.lower() in s.report_text.lower()]
            self.add_report(s.study_id, s.report_text, findings)
        self._built = True
        print(f"Medical RAG index built: {len(self.all_reports)} reports")

    def retrieve(
        self,
        predicted_diseases: list[str],
        k: int = 3,
    ) -> list[dict]:
        """Retrieve top-k reports most similar to predicted diseases.

        Similarity = number of shared diseases between prediction and report.
        """
        if not self._built:
            return []

        pred_set = set(predicted_diseases)
        scores: dict[int, int] = defaultdict(int)

        for d in pred_set:
            if d in self.disease_index:
                for idx, (study_id, text, findings_set) in enumerate(
                    self.disease_index[d]
                ):
                    # Find the index in all_reports
                    pass

        # Simpler: score each report by overlap with predicted diseases
        report_scores = []
        for i, rep in enumerate(self.all_reports):
            overlap = len(pred_set & set(rep["findings"]))
            if overlap > 0:
                report_scores.append((overlap, i))

        # Sort by overlap desc, take top-k
        report_scores.sort(key=lambda x: -x[0])
        top_k = report_scores[:k]

        results = []
        for overlap, i in top_k:
            rep = self.all_reports[i]
            results.append({
                "study_id": rep["study_id"],
                "report_text": rep["report_text"],
                "findings": rep["findings"],
                "overlap": overlap,
            })
        return results

    def retrieve_context_text(
        self,
        predicted_diseases: list[str],
        k: int = 3,
        max_chars_per_report: int = 400,
    ) -> str:
        """Build a text context string from retrieved reports."""
        results = self.retrieve(predicted_diseases, k=k)
        if not results:
            return ""

        lines = ["Retrieved similar clinical cases for reference:"]
        for r in results:
            snippet = r["report_text"][:max_chars_per_report]
            lines.append(
                f"- Case {r['study_id']} (shared diseases: {r['overlap']}): {snippet}"
            )
        lines.append("")
        return "\n".join(lines)


def build_rag_index_from_samples(samples) -> MedicalRAGIndex:
    rag = MedicalRAGIndex()
    rag.build(samples)
    return rag
