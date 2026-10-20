"""Evaluation module for Chest Xpert.

Computes per-label AUROC and AUPRC on a properly licensed, available test
dataset. Preserves patient-level separation where patient identifiers exist and
respects uncertain/masked labels.

STATUS: This script is ready to run but evaluation has NOT yet been performed.
No results are claimed until the script is executed on an actual dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from chest_xpert.config import LABELS_18, MODEL_CONFIG
from chest_xpert.inference import InferenceEngine
from chest_xpert.label_mapping import map_outputs_to_labels


@dataclass
class EvalResult:
    label: str
    auroc: Optional[float]
    auprc: Optional[float]
    n_pos: int
    n_neg: int
    n_uncertain: int
    n_total: int
    computable: bool  # False when both classes not present


@dataclass
class EvaluationReport:
    model_name: str
    dataset: str
    n_patients: int
    n_images: int
    results: List[EvalResult] = field(default_factory=list)
    run: bool = False
    note: str = ""


def _binary_mask(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split labels into positive/negative masks, treating uncertain as masked.

    Expects labels in {{0, 1, u}} where u marks uncertain/masked. Returns
    (y_true, valid_mask) with valid_mask False for uncertain entries.
    """
    y = labels.astype(float)
    valid = ~np.isnan(y)
    y_clean = np.where(valid, y, 0.0)
    return y_clean, valid


def evaluate_dataset(
    image_paths: List[str],
    labels: np.ndarray,          # shape [n_images, 18], values in {0,1,nan}
    patient_ids: Optional[List[str]] = None,
    model_name: str = MODEL_CONFIG.name,
    device: str = "cpu",
    batch_size: int = 1,
    dataset_name: str = "unspecified",
) -> EvaluationReport:
    """Run inference on a dataset and compute per-label AUROC/AUPRC.

    Patient-level separation: if patient_ids are provided, results are still
    computed per-image but the caller should ensure the dataset split keeps
    patients out of training. Here we only perform inference + metric
    computation on the supplied split.

    Uncertain/masked labels (nan) are excluded from metric computation for
    that label.
    """
    engine = InferenceEngine(model_name=model_name, device=device)
    report = EvaluationReport(
        model_name=model_name,
        dataset=dataset_name,
        n_images=len(image_paths),
        n_patients=len(set(patient_ids)) if patient_ids else 0,
        run=False,
        note=(
            "Evaluation not yet run. Execute evaluate_dataset() on a real, "
            "properly licensed test split to populate results."
        ),
    )

    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS_18)}
    scores_all: List[Dict[str, float]] = []

    for path in image_paths:
        from chest_xpert.preprocessing import preprocess
        pp = preprocess(path, device)
        result = engine.predict(pp.tensor)
        scores_all.append(result.scores)

    # Build per-label arrays.
    n = len(image_paths)
    for lbl in LABELS_18:
        col_idx = label_to_idx[lbl]
        y_raw = labels[:, col_idx]
        valid = ~np.isnan(y_raw)
        y_true = np.where(valid, y_raw, 0.0).astype(int)

        y_score = np.array([s.get(lbl, float("nan")) for s in scores_all], dtype=float)
        score_valid = valid & ~np.isnan(y_score)

        n_pos = int((y_true[score_valid] == 1).sum())
        n_neg = int((y_true[score_valid] == 0).sum())
        n_uncertain = int((~valid).sum())

        res = EvalResult(
            label=lbl,
            auroc=None,
            auprc=None,
            n_pos=n_pos,
            n_neg=n_neg,
            n_uncertain=n_uncertain,
            n_total=int(score_valid.sum()),
            computable=False,
        )

        if score_valid.sum() >= 10 and n_pos > 0 and n_neg > 0:
            try:
                res.auroc = float(roc_auc_score(y_true[score_valid], y_score[score_valid]))
                res.auprc = float(average_precision_score(y_true[score_valid], y_score[score_valid]))
                res.computable = True
            except ValueError:
                res.computable = False

        report.results.append(res)

    report.run = True
    report.note = "Evaluation completed on the supplied split."
    return report


def save_report(report: EvaluationReport, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "model_name": report.model_name,
        "dataset": report.dataset,
        "n_patients": report.n_patients,
        "n_images": report.n_images,
        "run": report.run,
        "note": report.note,
        "results": [
            {
                "label": r.label,
                "auroc": r.auroc,
                "auprc": r.auprc,
                "n_pos": r.n_pos,
                "n_neg": r.n_neg,
                "n_uncertain": r.n_uncertain,
                "n_total": r.n_total,
                "computable": r.computable,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
