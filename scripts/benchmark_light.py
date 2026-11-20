"""Lightweight benchmark: Stage A classifier + HTML report (no LLM downloads)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG, PROJECT_ROOT
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess, validate_image
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.report import ReportData, generate_html

SAMPLE_IMAGE = PROJECT_ROOT / "samples" / "sample_frontal.png"


def main() -> None:
    print("=" * 55)
    print("Chest Xpert — Stage A + HTML report latency")
    print("=" * 55)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE_CONFIG.resolve()}")
    print(f"Sample: {SAMPLE_IMAGE}")
    print("-" * 55)

    total_start = time.perf_counter()

    # Stage A
    t0 = time.perf_counter()
    engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=DEVICE_CONFIG.resolve())
    pp = preprocess(str(SAMPLE_IMAGE), DEVICE_CONFIG.resolve())
    result = engine.predict(pp.tensor)
    mappings = map_outputs_to_labels(result.raw, result.model_name)
    stage_a_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Stage A] classifier: {stage_a_ms:.1f} ms (inference only: {result.inference_only_ms:.1f} ms)")
    print(f"  device={result.device} supported={len(result.supported_labels)}/18")

    # Stage C/D HTML
    t0 = time.perf_counter()
    rd = ReportData(
        model_name=result.model_name,
        model_version="pre-trained weights (torchxrayvision)",
        device=result.device,
        latency_ms=result.latency_ms,
        inference_only_ms=result.inference_only_ms,
        mappings=mappings,
        unsupported_labels=result.unsupported_labels,
    )
    html = generate_html(rd)
    stage_cd_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Stage C/D] HTML report: {stage_cd_ms:.1f} ms")

    total_ms = (time.perf_counter() - total_start) * 1000.0
    print("-" * 55)
    print(f"TOTAL (Stage A + C/D): {total_ms:.1f} ms ({total_ms/1000:.2f} s)")
    print("=" * 55)

    # Show a sample of the report
    print("\n--- Report preview (first 1500 chars) ---")
    # Strip HTML tags roughly for display
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    print(text[:1500])


if __name__ == "__main__":
    main()
