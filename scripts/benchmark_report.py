"""Benchmark: measure end-to-end time for one Chest Xpert report.

Runs the full pipeline (Stage 0 -> Stage A -> Stage B -> HTML report) on the
bundled sample image and prints per-stage latency.

Usage:
    source /home/doli/anaconda3/bin/activate hct-lsr
    export CHESTXPERT_HF_TOKEN="your_token"   # optional, for Qwen VLM stages
    python scripts/benchmark_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chest_xpert.config import (
    DEVICE_CONFIG,
    LLM_MODEL_CONFIG,
    MODALITY_CONFIG,
    MODEL_CONFIG,
    PROJECT_ROOT,
)
from chest_xpert.inference import InferenceEngine
from chest_xpert.modality import ModalityValidator
from chest_xpert.preprocessing import preprocess, validate_image
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.report import ReportData, generate_html
from chest_xpert.verification import VerificationEngine, LLMModelClient


SAMPLE_IMAGE = PROJECT_ROOT / "samples" / "sample_frontal.png"


def t0() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0  # ms


def main() -> None:
    print("=" * 60)
    print("Chest Xpert — per-report latency benchmark")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolved device (classifier): {DEVICE_CONFIG.resolve()}")
    print(f"LLM backend: {LLM_MODEL_CONFIG.backend}")
    print(f"LLM model: {LLM_MODEL_CONFIG.model_id}")
    print(f"Load in 4-bit: {LLM_MODEL_CONFIG.load_in_4bit}")
    print(f"Force CPU: {LLM_MODEL_CONFIG.force_cpu}")
    print(f"Sample image: {SAMPLE_IMAGE} ({SAMPLE_IMAGE.stat().st_size / 1024:.1f} KB)")
    print("-" * 60)

    total_start = t0()

    # ------------------------------------------------------------------
    # Stage 0 — Image & modality validation
    # ------------------------------------------------------------------
    print("\n[Stage 0] Image & modality validation...")
    stage0_start = t0()

    validation = validate_image(str(SAMPLE_IMAGE))
    print(f"  File validation: {'OK' if validation.ok else 'FAILED'} ({validation.size_mb:.2f} MB, {validation.dims})")
    if not validation.ok:
        print("  Aborting — file validation failed.")
        return

    validator = ModalityValidator()
    modality_result = validator.validate(str(SAMPLE_IMAGE))
    print(f"  Modality decision: {modality_result.decision}")
    print(f"  Modality method:   {modality_result.method}")
    print(f"  Modality model:    {modality_result.model_id}")
    print(f"  Stage 0 latency:   {elapsed(stage0_start):.1f} ms")

    # ------------------------------------------------------------------
    # Stage A — Disease classification
    # ------------------------------------------------------------------
    print("\n[Stage A] Disease classification (torchxrayvision DenseNet-121)...")
    stage_a_start = t0()

    device_resolved = DEVICE_CONFIG.resolve()
    engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=device_resolved)
    pp = preprocess(str(SAMPLE_IMAGE), device_resolved)
    result = engine.predict(pp.tensor)
    mappings = map_outputs_to_labels(result.raw, result.model_name)

    print(f"  Device:            {result.device}")
    print(f"  Model:             {result.model_name}")
    print(f"  Supported labels:  {len(result.supported_labels)}/18")
    print(f"  Stage A latency:   {elapsed(stage_a_start):.1f} ms")
    print(f"    (inference only: {result.inference_only_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # Stage B — LLM verification + V7 report
    # ------------------------------------------------------------------
    print("\n[Stage B] LLM verification + V7 report generation...")
    stage_b_start = t0()

    client = LLMModelClient(
        model_id=LLM_MODEL_CONFIG.model_id,
        backend=LLM_MODEL_CONFIG.backend,
        force_cpu=LLM_MODEL_CONFIG.force_cpu,
        vllm_url=LLM_MODEL_CONFIG.vllm_url,
        load_in_4bit=LLM_MODEL_CONFIG.load_in_4bit,
    )
    print(f"  LLM configured: {client.is_configured()}")
    if client.is_configured():
        model_doc = (
            "torchxrayvision DenseNet-121 (densenet121-res224-all), trained by "
            "other researchers on mixed public chest X-ray datasets. Scores are raw "
            "model outputs, not calibrated probabilities."
        )
        vr = client.generate_v7_report(
            mappings=mappings,
            model_doc=model_doc,
            patient_info={},
            image_path=str(SAMPLE_IMAGE),
            contact_sheet_path=None,
            few_shot_examples=None,
            self_correct=False,  # single pass for timing
        )
        print(f"  Verification available: {vr.available}")
        print(f"  LLM model:              {vr.model_id}")
        print(f"  Stage B latency:        {elapsed(stage_b_start):.1f} ms")
        if vr.error:
            print(f"  Error:                  {vr.error}")
    else:
        print("  LLM not configured — skipping Stage B (set CHESTXPERT_HF_TOKEN).")
        print("  Stage B latency:        N/A (skipped)")

    # ------------------------------------------------------------------
    # Stage C/D — HTML report generation (template-based, fast)
    # ------------------------------------------------------------------
    print("\n[Stage C/D] HTML report generation...")
    stage_cd_start = t0()

    rd = ReportData(
        model_name=result.model_name,
        model_version="pre-trained weights (torchxrayvision)",
        device=result.device,
        latency_ms=result.latency_ms,
        inference_only_ms=result.inference_only_ms,
        modality_ms=elapsed(stage0_start),
        mappings=mappings,
        unsupported_labels=result.unsupported_labels,
    )
    html = generate_html(rd)
    print(f"  Stage C/D latency:    {elapsed(stage_cd_start):.1f} ms")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_ms = elapsed(total_start)
    print("\n" + "=" * 60)
    print("SUMMARY — end-to-end time for one report")
    print("=" * 60)
    print(f"  Stage 0 (modality):   {elapsed(stage0_start):.1f} ms")
    print(f"  Stage A (classifier): {elapsed(stage_a_start):.1f} ms")
    if client.is_configured():
        print(f"  Stage B (LLM):        {elapsed(stage_b_start):.1f} ms")
    else:
        print(f"  Stage B (LLM):        skipped (no token)")
    print(f"  Stage C/D (HTML):     {elapsed(stage_cd_start):.1f} ms")
    print(f"  --------------------------------")
    print(f"  TOTAL:                {total_ms:.1f} ms ({total_ms/1000:.2f} s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
