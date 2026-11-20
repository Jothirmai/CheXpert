"""Generate a full report HTML file from the sample image (Stage A only, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG, PROJECT_ROOT
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.report import ReportData, generate_html, generate_pdf

SAMPLE_IMAGE = PROJECT_ROOT / "samples" / "sample_frontal.png"
OUT_HTML = PROJECT_ROOT / "reports" / "sample_report.html"
OUT_PDF = PROJECT_ROOT / "reports" / "sample_report.pdf"


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    device = DEVICE_CONFIG.resolve()
    engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=device)
    pp = preprocess(str(SAMPLE_IMAGE), device)
    result = engine.predict(pp.tensor)
    mappings = map_outputs_to_labels(result.raw, result.model_name)

    rd = ReportData(
        model_name=result.model_name,
        model_version="pre-trained weights (torchxrayvision)",
        device=result.device,
        latency_ms=result.latency_ms,
        inference_only_ms=result.inference_only_ms,
        mappings=mappings,
        unsupported_labels=result.unsupported_labels,
    )
    rd.exam.patient_id = "PAT-001"
    rd.exam.exam_date = "2024-01-15"
    rd.exam.view = "PA"
    rd.impression = "Draft impression pending radiologist review."

    html = generate_html(rd)
    OUT_HTML.write_text(html)
    print(f"HTML report saved to: {OUT_HTML}")
    print(f"Size: {OUT_HTML.stat().st_size / 1024:.1f} KB")

    try:
        generate_pdf(rd, OUT_PDF)
        print(f"PDF report saved to: {OUT_PDF}")
        print(f"Size: {OUT_PDF.stat().st_size / 1024:.1f} KB")
    except Exception as exc:
        print(f"PDF generation skipped: {exc}")

    # Print all 18 label scores
    print("\n--- All 18 label scores ---")
    for m in mappings:
        if m.supported:
            print(f"  {m.requested_label:30s}: {m.score:.4f}")
        else:
            print(f"  {m.requested_label:30s}: unsupported")


if __name__ == "__main__":
    main()
