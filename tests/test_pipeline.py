"""Tests for Stage 0 modality, Stage B verification, Stage C report, and pipeline no-leak."""

from __future__ import annotations

from pathlib import Path

import pytest

from chest_xpert.config import LABELS_18, MODALITY_CONFIG, MODEL_CONFIG
from chest_xpert.label_mapping import LabelMapping, map_outputs_to_labels
from chest_xpert.modality import ModalityDecision, ModalityResult, ModalityValidator
from chest_xpert.report import generate_html, generate_pdf, ReportData
from chest_xpert.verification import (
    Assessment,
    VerificationEngine,
    VerificationItem,
    VerificationResult,
    _validate_and_build_items,
)


def _mapping(label: str, score: float, supported: bool = True) -> LabelMapping:
    return LabelMapping(
        requested_label=label,
        supported=supported,
        model_output=label if supported else None,
        score=score if supported else None,
    )


# ---------------- Modality decision logic ----------------

def test_modality_result_accepted():
    r = ModalityResult(
        decision=ModalityDecision.ACCEPTED,
        chest_xray_prob=0.8,
        best_other_label="a photograph",
        best_other_prob=0.1,
        reason="ok",
        model_id="clip",
    )
    assert r.is_accepted and not r.is_rejected and not r.is_uncertain


def test_modality_result_rejected():
    r = ModalityResult(
        decision=ModalityDecision.REJECTED,
        chest_xray_prob=0.1,
        best_other_label="a photograph",
        best_other_prob=0.8,
        reason="not xray",
        model_id="clip",
    )
    assert r.is_rejected


def test_modality_result_uncertain():
    r = ModalityResult(
        decision=ModalityDecision.UNCERTAIN,
        chest_xray_prob=0.4,
        best_other_label="a photograph",
        best_other_prob=0.35,
        reason="uncertain",
        model_id="clip",
    )
    assert r.is_uncertain


def test_modality_result_technical_failure_not_rejection():
    r = ModalityResult(
        decision=ModalityDecision.TECHNICAL_FAILURE,
        chest_xray_prob=None,
        reason="model load failed",
        technical_error="download error",
        model_id="clip",
    )
    assert r.is_technical_failure
    assert not r.is_rejected


def test_validator_disabled_returns_technical_failure():
    v = ModalityValidator()
    v.config.enabled = False
    r = v.validate("/tmp/nonexistent.png")
    assert r.is_technical_failure
    assert r.decision == ModalityDecision.TECHNICAL_FAILURE


# ---------------- Verification schema validation ----------------

def test_validation_accepts_valid_items():
    mappings = [_mapping("Atelectasis", 0.9), _mapping("Cardiomegaly", 0.2)]
    parsed = {
        "image_assessment": "Image appears interpretable.",
        "items": [
            {
                "pathology": "Atelectasis",
                "classifier_score": 0.9,
                "assessment": "consistent",
                "rationale": "Supports opacity.",
                "uncertainty": "Low.",
                "human_review_recommended": True,
            },
            {
                "pathology": "Cardiomegaly",
                "classifier_score": 0.2,
                "assessment": "uncertain",
                "rationale": "Borderline.",
                "uncertainty": "Moderate.",
                "human_review_recommended": True,
            },
        ],
    }
    items, findings, warnings = _validate_and_build_items(parsed, mappings)
    assert len(items) == 2
    assert items[0].assessment == Assessment.CONSISTENT
    assert items[0].classifier_score == 0.9
    assert items[1].assessment == Assessment.UNCERTAIN
    assert warnings == []


def test_validation_skips_unsupported_pathology():
    mappings = [_mapping("Atelectasis", 0.9)]
    parsed = {
        "items": [
            {"pathology": "NotARealPathology", "assessment": "consistent",
             "rationale": "x", "uncertainty": "x", "human_review_recommended": False},
        ]
    }
    items, findings, warnings = _validate_and_build_items(parsed, mappings)
    assert items == []
    assert any("NotARealPathology" in w for w in warnings)


def test_validation_defaults_invalid_assessment_to_uncertain():
    mappings = [_mapping("Atelectasis", 0.9)]
    parsed = {
        "items": [
            {"pathology": "Atelectasis", "assessment": "nonsense",
             "rationale": "x", "uncertainty": "x", "human_review_recommended": False}
        ]
    }
    items, findings, warnings = _validate_and_build_items(parsed, mappings)
    assert items[0].assessment == Assessment.UNCERTAIN


# ---------------- VerificationResult helpers ----------------

def test_verification_result_disagreements_and_uncertain():
    items = [
        VerificationItem("Atelectasis", 0.9, "consistent", "r", "u", True),
        VerificationItem("Cardiomegaly", 0.2, "inconsistent", "r", "u", True),
        VerificationItem("Edema", 0.5, "uncertain", "r", "u", True),
    ]
    vr = VerificationResult(items=items, available=True, model_id="llm")
    assert len(vr.disagreements()) == 1
    assert len(vr.uncertain_items()) == 1
    assert vr.item_for("Atelectasis") is not None
    assert vr.item_for("Missing") is None


# ---------------- Report with verification ----------------

def test_report_html_includes_verification_and_modality():
    mappings = [_mapping(l, 0.5) for l in LABELS_18[:3]]
    vr = VerificationResult(
        items=[
            VerificationItem("Atelectasis", 0.5, "consistent", "ok", "low", True),
        ],
        available=True,
        model_id="llm-x",
    )
    rd = ReportData()
    rd.model_name = MODEL_CONFIG.name
    rd.modality_decision = "accepted"
    rd.modality_reason = "looks like CXR"
    rd.modality_model = "clip"
    rd.modality_ms = 123.0
    rd.mappings = mappings
    rd.unsupported_labels = []
    rd.verification_result = vr
    rd.verification_available = True
    rd.verification_ms = 456.0
    rd.impression = "draft impression"
    rd.reviewer_name = "Dr. X"
    rd.review_status = "Pending"

    html = generate_html(rd)
    assert "Stage 0 — Image &amp; Modality Validation" in html
    assert "accepted" in html
    assert "Stage B — Prediction Verification (LLM #1)" in html
    assert "llm-x" in html
    assert "consistent" in html
    assert "AI-generated research draft" in html
    assert "radiologist" in html


def test_report_pdf_with_verification(tmp_path):
    mappings = [_mapping("Atelectasis", 0.5)]
    rd = ReportData()
    rd.model_name = MODEL_CONFIG.name
    rd.mappings = mappings
    rd.unsupported_labels = []
    rd.verification_result = VerificationResult(available=False, error="not configured")
    out = tmp_path / "r.pdf"
    generate_pdf(rd, out)
    assert out.exists() and out.stat().st_size > 0


# ---------------- Label mapping round-trip ----------------

def test_map_outputs_to_labels_returns_all_18():
    raw = {"Atelectasis": 0.9}
    mappings = map_outputs_to_labels(raw, MODEL_CONFIG.name)
    assert [m.requested_label for m in mappings] == LABELS_18
    assert all(m.supported for m in mappings)
