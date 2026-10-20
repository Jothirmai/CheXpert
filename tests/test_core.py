"""Tests for Chest Xpert core modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chest_xpert.config import LABELS_18, MODEL_CONFIG
from chest_xpert.label_mapping import map_outputs_to_labels, supported_labels, unsupported_labels
from chest_xpert.model_registry import registry
from chest_xpert.preprocessing import preprocess, validate_image
from chest_xpert.report import generate_html, generate_pdf, ReportData


SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_frontal.png"


def test_model_supports_all_18_labels():
    info = registry()[MODEL_CONFIG.name]
    supported = [ls for ls in info.supported_labels if ls.supported]
    assert len(supported) == len(LABELS_18) == 18
    # Every requested label has an entry.
    assert {ls.requested_label for ls in info.supported_labels} == set(LABELS_18)


def test_no_unsupported_labels_for_all_model():
    assert unsupported_labels(MODEL_CONFIG.name) == []
    assert len(supported_labels(MODEL_CONFIG.name)) == 18


def test_label_mapping_preserves_order_and_marks_unsupported():
    raw = {"Atelectasis": 0.9, "Cardiomegaly": 0.2}
    mappings = map_outputs_to_labels(raw, MODEL_CONFIG.name)
    assert [m.requested_label for m in mappings] == LABELS_18
    atelectasis = next(m for m in mappings if m.requested_label == "Atelectasis")
    assert atelectasis.supported is True
    assert atelectasis.score == 0.9
    # Labels not in raw but supported by the model still appear with score None.
    edema = next(m for m in mappings if m.requested_label == "Edema")
    assert edema.supported is True
    assert edema.score is None


def test_scores_dict_returns_all_18():
    mappings = map_outputs_to_labels({}, MODEL_CONFIG.name)
    d = {m.requested_label: m.score for m in mappings}
    assert set(d.keys()) == set(LABELS_18)


def test_preprocess_output_shape_and_range():
    pp = preprocess(SAMPLE, "cpu", resolution=MODEL_CONFIG.input_resolution)
    assert pp.tensor.shape == (1, 1, MODEL_CONFIG.input_resolution, MODEL_CONFIG.input_resolution)
    # torchxrayvision normalization -> roughly [-1024, 1024]
    assert pp.tensor.min().item() >= -1025
    assert pp.tensor.max().item() <= 1025


def test_validation_rejects_bad_extension():
    bad = Path("/tmp/chest_xpert_test_bad.txt")
    bad.write_text("not an image")
    res = validate_image(bad)
    assert res.ok is False
    bad.unlink(missing_ok=True)


def test_report_generation_html_contains_disclaimer():
    rd = ReportData()
    rd.model_name = MODEL_CONFIG.name
    rd.mappings = map_outputs_to_labels({}, MODEL_CONFIG.name)
    rd.unsupported_labels = []
    html = generate_html(rd)
    assert "AI-generated research draft" in html
    assert "radiologist" in html
    assert "not calibrated probabilities" in html.lower()


def test_report_generation_pdf(tmp_path):
    rd = ReportData()
    rd.model_name = MODEL_CONFIG.name
    rd.mappings = map_outputs_to_labels({}, MODEL_CONFIG.name)
    out = tmp_path / "report.pdf"
    generate_pdf(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_failure_handling_invalid_label():
    mappings = map_outputs_to_labels({}, MODEL_CONFIG.name)
    # No unsupported labels for this model.
    unsupported = [m for m in mappings if not m.supported]
    assert unsupported == []
