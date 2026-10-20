"""Label mapping between model outputs and the 18 requested labels.

Provides a strict, auditable mapping: a requested label is only reported when
the model genuinely emits a matching output. Unsupported labels are listed
explicitly and are never derived from unrelated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from chest_xpert.config import LABELS_18
from chest_xpert.model_registry import registry, LabelSupport


@dataclass
class LabelMapping:
    requested_label: str
    supported: bool
    model_output: Optional[str]
    score: Optional[float]


def map_outputs_to_labels(
    raw_scores: Dict[str, float],
    model_name: str,
) -> List[LabelMapping]:
    """Map raw model outputs to the 18 requested labels.

    Args:
        raw_scores: model output name -> score, as returned by the model.
        model_name: key into the model registry.

    Returns:
        One LabelMapping per requested label, in the canonical 18-label order.
        Unsupported labels have ``supported=False`` and ``score=None``.
    """
    info = registry()[model_name]
    ls_by_requested = {ls.requested_label: ls for ls in info.supported_labels}

    mappings: List[LabelMapping] = []
    for requested in LABELS_18:
        ls = ls_by_requested.get(requested)
        if ls is None or not ls.supported:
            mappings.append(
                LabelMapping(
                    requested_label=requested,
                    supported=False,
                    model_output=None,
                    score=None,
                )
            )
            continue
        score = raw_scores.get(ls.model_output_name)
        mappings.append(
            LabelMapping(
                requested_label=requested,
                supported=True,
                model_output=ls.model_output_name,
                score=(
                    float(score) if score is not None else None
                ),
            )
        )
    return mappings


def supported_labels(model_name: str) -> List[str]:
    return [m.requested_label for m in map_outputs_to_labels({}, model_name) if m.supported]


def unsupported_labels(model_name: str) -> List[str]:
    return [m.requested_label for m in map_outputs_to_labels({}, model_name) if not m.supported]


def scores_dict(mappings: List[LabelMapping]) -> Dict[str, Optional[float]]:
    """Return {requested_label: score_or_None} for all 18 labels."""
    return {m.requested_label: m.score for m in mappings}
