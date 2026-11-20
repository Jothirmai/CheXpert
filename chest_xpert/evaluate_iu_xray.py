"""CliQ-RRG / Chest Xpert evaluation on IU X-Ray (Demner-Fushman et al., 2015).

Uses the existing torchxrayvision DenseNet-121 (18 diseases) classifier from
chest_xpert.inference + V7-format LLM report generation from
chest_xpert.verification.

For each image in the IU X-Ray test split:
  1. Preprocess (224x224 grayscale, normalize to [-1024, 1024]).
  2. Run DenseNet-121 classifier -> 18 raw scores.
  3. Generate Grad-CAM heatmaps for all supported findings + a contact sheet.
  4. Generate V7-format radiology report via LLM
     (Qwen/Qwen3-VL-32B-Instruct, vision-language on GPU;
      original X-ray + Grad-CAM contact sheet + classifier scores as input).
  5. Compare generated V7 report against the IU X-Ray reference report using
     NLG + clinical metrics.

All metrics are saved to a single JSON file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage

# Ensure project root on path so `chest_xpert` and `cliq_rrg` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chest_xpert.config import MODEL_CONFIG, LABELS_18
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess, load_image
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.explainability import (
    GradCAM,
    generate_contact_sheet,
    save_contact_sheet,
)
from chest_xpert.verification import (
    VerificationEngine, LLMModelClient,
)
from cliq_rrg.eval import (
    bleu, meteor, rouge_l, cider, bert_score,
    chexbert_metrics, radgraph_f1, evaluate_reports,
)
from cliq_rrg.dataset import IUXRayDownloader, parse_reports, build_samples, train_val_test_split, IU_FINDINGS


# ---------------------------------------------------------------------------
# IU X-Ray helpers
# ---------------------------------------------------------------------------

def extract_findings_from_report(report_text: str) -> list[str]:
    """Extract disease findings mentioned in an IU X-Ray report (free text)."""
    tl = report_text.lower()
    found = []
    for finding in IU_FINDINGS:
        if finding.lower() in tl:
            found.append(finding)
    return found


def clean_report_for_metrics(text: str) -> str:
    """Strip publisher / boilerplate junk from a report so NLG metrics compare
    clinical text to clinical text (not author names, DOIs, file paths)."""
    import re
    t = text or ""
    # Remove URLs
    t = re.sub(r"http[s]?://[^\s]+", " ", t)
    # Remove standalone hash fragments
    t = re.sub(r"\b\d+k\d+\b", " ", t)
    # Remove file paths
    t = re.sub(r"/hadoop/storage[^\s]*", " ", t)
    # Remove author name pattern "First Last MD"
    t = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+ MD\b", " ", t)
    # Remove stray years (4-digit standalone)
    t = re.sub(r"(?<!\d)\d{4}(?!\d)", " ", t)
    # Remove "F1"/"F2" view markers
    t = re.sub(r"\bF[12]\b", " ", t)

    # Sentence-level filter: drop sentences that are pure boilerplate.
    # Split on sentence boundaries but keep clinical sentences.
    sentences = re.split(r"(?<=[.!?])\s+", t)
    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        sl = s.lower()
        # Drop pure boilerplate sentences.
        if any(tok in sl for tok in [
            "cxr open-access",
            "byncnd",
            "creativecommons",
            "indiana university chest x-ray collection",
            "the data are drawn from multiple hospital",
            "the data are drawn from multiple hospitals",
            "radiology report",
            "eng radiology report",
            "pa and lateral",
            "normal f1",
        ]):
            continue
        # Drop very short fragments (< 3 chars) that are just noise.
        if len(s) < 3:
            continue
        kept.append(s)
    t = " ".join(kept)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def retrieve_few_shot(
    current_probs: dict,
    per_sample: list[dict],
    k: int = 3,
) -> list[dict]:
    """Retrieve top-k previously processed cases whose classifier score
    profile is most similar to the current case (cosine similarity).
    Returns dicts with study_id and clean_report for few-shot prompting."""
    if not per_sample:
        return []

    import numpy as np

    # Build vector for current case (fixed label order = LABELS_18).
    cur_vec = np.array(
        [float(current_probs.get(l, 0.0)) for l in LABELS_18], dtype=np.float32
    )
    cur_norm = np.linalg.norm(cur_vec)
    if cur_norm == 0:
        cur_norm = 1e-8

    scored = []
    for ps in per_sample:
        ps_probs = ps.get("classifier_scores", {}) or {}
        ps_vec = np.array(
            [float(ps_probs.get(l, 0.0)) for l in LABELS_18], dtype=np.float32
        )
        ps_norm = np.linalg.norm(ps_vec)
        if ps_norm == 0:
            ps_norm = 1e-8
        sim = float(np.dot(cur_vec, ps_vec) / (cur_norm * ps_norm))
        scored.append((sim, ps))

    scored.sort(key=lambda x: x[0], reverse=True)

    few_shot = []
    for sim, ps in scored[:k]:
        few_shot.append({
            "study_id": ps.get("study_id", "?"),
            "clean_report": clean_report_for_metrics(
                ps.get("generated_report", "")
            )[:700],
        })
    return few_shot


def load_iu_xray_split(data_root: str | Path = "./data/iu_xray",
                       split: str = "test",
                       max_samples: Optional[int] = None):
    """Load an IU X-Ray split ('train', 'val', or 'test') as list of dicts."""
    downloader = IUXRayDownloader(root=data_root)
    downloader.download_all(force=False)

    reports = parse_reports(downloader.reports_dir)
    samples = build_samples(downloader.images_dir, reports)

    train_s, val_s, test_s = train_val_test_split(samples, seed=42)
    split_map = {"train": train_s, "val": val_s, "test": test_s}
    items_s = split_map.get(split, test_s)

    if max_samples is not None:
        items_s = items_s[:max_samples]

    items = []
    for s in items_s:
        items.append({
            "image_path": s.image_path,
            "report_text": s.report_text,
            "study_id": s.study_id,
            "view": s.view,
        })
    return items


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_iu_xray(
    data_root: str | Path = "./data/iu_xray",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    split: str = "test",
    llm_model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
    finetuned_model_dir: Optional[str | Path] = None,
    llm_api_url: Optional[str] = None,
    llm_api_token: Optional[str] = None,
    output_file: str | Path = "./checkpoints/iu_xray_evaluation_metrics.json",
) -> dict:
    """Run full evaluation: classifier + V7 LLM report + metrics on IU X-Ray."""

    print(f"Device: {device}")
    print(f"IU X-Ray root: {data_root}")
    print(f"Split: {split}")
    llm_display = str(finetuned_model_dir or llm_model_id)
    print(f"LLM: {llm_display} (api_url={llm_api_url})")
    print(f"Max samples: {max_samples or 'all'}")
    print()

    # ---------- Load dataset ----------
    items = load_iu_xray_split(data_root=data_root, split=split, max_samples=max_samples)
    print(f"{split.capitalize()} samples: {len(items)}")

    # ---------- Load classifier (torchxrayvision DenseNet-121, 18 diseases) ----------
    engine = InferenceEngine(device=device)
    _ = engine.model  # trigger load
    print(f"Classifier loaded: {engine.model_name} on {engine.device}")

    # --- Calibrated thresholds (torchxrayvision op_threshs per label) ---
    op_threshs: dict = {}
    try:
        op_threshs = dict(engine.model.op_threshs)  # label -> threshold
    except Exception:
        op_threshs = {}
    print(f"Calibrated thresholds loaded: {len(op_threshs)} labels")

    # ---------- Load LLM for V7 report generation ----------
    # Use fine-tuned model if provided, otherwise base model_id
    effective_model_id = str(finetuned_model_dir) if finetuned_model_dir else llm_model_id
    llm_client = LLMModelClient(
        model_id=effective_model_id,
        backend="transformers_local",
        force_cpu=(device == "cpu"),
        load_in_4bit=True,
    )
    ve = VerificationEngine(client=llm_client)
    print(f"LLM configured: {llm_client.is_configured()}")
    print()

    # ---------- Grad-CAM engine (reused across samples) ----------
    gradcam = GradCAM(engine.model, target_layer_name="features.norm5")

    # ---------- Model description (for output file) ----------
    from chest_xpert.model_registry import registry
    model_info = registry()[MODEL_CONFIG.name]

    # ---------- Run evaluation ----------
    references = []
    candidates = []
    per_sample = []
    errors = 0

    # Resume support: load already-processed study IDs if output exists.
    out_path = Path(output_file)
    processed_ids: set[str] = set()
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
            for ps in existing.get("per_sample", []):
                processed_ids.add(ps["study_id"])
            references = existing.get("references", [])
            candidates = existing.get("candidates", [])
            per_sample = existing.get("per_sample", [])
            print(f"Resuming: {len(processed_ids)} samples already processed.")
        except Exception:  # noqa: BLE001
            processed_ids = set()

    model_doc = (
        "torchxrayvision DenseNet-121 (densenet121-res224-all). "
        "Multi-label chest X-ray classifier trained on NIH ChestX-ray14, "
        "PadChest, CheXpert, MIMIC-CXR, ChestX-ray8, Google, OpenI, Kaggle/RSNA. "
        "Outputs are independent per-label logits (not calibrated probabilities). "
        "Input: 224x224 single-channel grayscale normalized to [-1024, 1024]."
    )

    for idx, item in enumerate(items):
        image_path = item["image_path"]
        reference_report = item["report_text"]
        study_id = item["study_id"]

        if study_id in processed_ids:
            print(f"  SKIP {study_id}: already processed (resume).")
            continue

        try:
            # --- Preprocess ---
            pp = preprocess(image_path, device, resolution=MODEL_CONFIG.input_resolution)

            # --- Classify ---
            result = engine.predict(pp.tensor)

            # --- Map to 18 labels ---
            mappings = map_outputs_to_labels(result.raw, engine.model_name)

            # --- Build patient info ---
            patient_info = {
                "patient_id": study_id,
                "study": "Chest X-ray",
                "view": item["view"] or "Posteroanterior (PA)",
                "patient_age": "NOT PROVIDED",
                "patient_sex": "NOT PROVIDED",
                "clinical_indication": "NOT PROVIDED",
                "comparison_study": "NOT PROVIDED",
            }

            # --- Disease names + probs from supported mappings ---
            disease_names = [
                m.requested_label for m in mappings if m.supported
            ]
            disease_probs = {
                m.requested_label: (m.score if m.score is not None else 0.0)
                for m in mappings if m.supported
            }

            # --- Skip empty reference reports ---
            if not reference_report or len(reference_report.strip()) < 20:
                errors += 1
                print(f"  SKIP {study_id}: empty reference report")
                continue

            # --- Grad-CAM contact sheet for all supported findings ---
            contact_sheet_path = None
            try:
                orig = load_image(image_path)  # [H, W] in [-1024, 1024]
                orig_disp = np.clip((orig + 1024) / 2048.0, 0.0, 1.0)
                cam_results = []
                for m in mappings:
                    if not m.supported:
                        continue
                    try:
                        g = gradcam.generate(
                            input_tensor=pp.tensor,
                            target_label=m.requested_label,
                            model_outputs=result.model_outputs,
                            original_image=orig_disp,
                        )
                        cam_results.append(g)
                    except Exception:  # noqa: BLE001
                        continue
                if cam_results:
                    sheet = generate_contact_sheet(cam_results, orig_disp)
                    contact_sheet_path = str(
                        Path(tempfile.mkdtemp()) / "contact_sheet.png"
                    )
                    save_contact_sheet(sheet, contact_sheet_path)
            except Exception:  # noqa: BLE001
                contact_sheet_path = None

            # --- Few-shot RAG: retrieve similar previously-processed cases ---
            few_shot = retrieve_few_shot(disease_probs, per_sample, k=3)

            # --- Generate V7 report via LLM (image + contact sheet + scores) ---
            vr = ve.generate_v7_report(
                mappings=mappings,
                model_doc=model_doc,
                patient_info=patient_info,
                image_path=image_path,
                contact_sheet_path=contact_sheet_path,
                few_shot_examples=few_shot,
                self_correct=True,
            )

            if not vr.available or not vr.v7_report:
                errors += 1
                print(f"  SKIP {study_id}: LLM unavailable ({vr.error})")
                continue

            generated_report = vr.v7_report

            # --- Clean reports for NLG metrics (strip publisher boilerplate) ---
            clean_ref = clean_report_for_metrics(reference_report)
            clean_gen = clean_report_for_metrics(generated_report)

            # --- Metrics per sample (on cleaned text) ---
            b = bleu(clean_ref, clean_gen)
            m_ = meteor(clean_ref, clean_gen)
            r = rouge_l(clean_ref, clean_gen)
            c = cider([clean_ref], clean_gen)
            bs = bert_score(clean_ref, clean_gen)

            ref_findings = extract_findings_from_report(reference_report)
            # Calibrated thresholds per label (fallback 0.5 if missing).
            pred_findings = [
                d for d, p in disease_probs.items()
                if p > op_threshs.get(d, 0.5)
            ]
            cx = chexbert_metrics(pred_findings, ref_findings)
            rg = radgraph_f1(
                [(d, "present") for d in pred_findings],
                [(d, "present") for d in ref_findings],
            )

            references.append(reference_report)
            candidates.append(generated_report)

            per_sample.append({
                "study_id": study_id,
                "image_path": image_path,
                "bleu": b,
                "meteor": m_,
                "rouge_l": r,
                "cider": c,
                "bert_score": bs,
                "chexbert": cx,
                "radgraph_f1": rg,
                "classifier_scores": disease_probs,
                "predicted_findings": pred_findings,
                "reference_findings": ref_findings,
                "generated_report": generated_report,
                "reference_report": reference_report,
                "v7_latency_ms": vr.latency_ms,
            })

            # --- Incremental save after each sample (resume-safe) ---
            agg_now = evaluate_reports(references, candidates) if references else {}
            overall_now = {
                "timestamp": datetime.now().isoformat(),
                "dataset": "IU X-Ray (Demner-Fushman et al., 2015)",
                "test_split": f"{split} split (7:2:1 split, seed=42)",
                "num_samples": len(references),
                "num_errors": errors,
                "classifier": {
                    "model_name": engine.model_name,
                    "architecture": model_info.architecture,
                    "parameter_count": model_info.parameter_count,
                    "authors": model_info.authors,
                    "organization": model_info.organization,
                    "publication": model_info.publication,
                    "repository": model_info.repository,
                    "training_datasets": model_info.training_datasets,
                    "training_objective": model_info.training_objective,
                    "loss_function": model_info.loss_function,
                    "input_dimensions": model_info.input_dimensions,
                    "preprocessing": model_info.preprocessing,
                    "multi_label_head": model_info.multi_label_head,
                    "license": model_info.license,
                    "permitted_uses": model_info.permitted_uses,
                    "limitations": model_info.limitations,
                    "evaluation_datasets": model_info.evaluation_datasets,
                    "published_metrics": model_info.published_metrics,
                    "weights_url": model_info.weights_url,
                    "trained_by_this_project": model_info.trained_by_this_project,
                    "supported_labels": [
                        {"requested_label": ls.requested_label, "supported": ls.supported,
                         "model_output_name": ls.model_output_name}
                        for ls in model_info.supported_labels
                    ],
                },
                "llm": {
                    "model_id": effective_model_id,
                    "base_model_id": llm_model_id,
                    "finetuned": finetuned_model_dir is not None,
                    "backend": "transformers_local",
                    "device": device,
                    "configured": llm_client.is_configured(),
                },
                "metrics": agg_now,
                "per_sample": per_sample,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(overall_now, f, indent=2)

            # --- Incremental text sidecar (regenerate fully each flush) ---
            txt_path = out_path.with_suffix(".txt")
            with open(txt_path, "w") as tf:
                tf.write("=" * 80 + "\n")
                tf.write("CHEST X-PERT — V7 GENERATED vs ORIGINAL REPORT COMPARISON\n")
                tf.write("=" * 80 + "\n\n")
                tf.write(f"Split: {split}\n")
                tf.write(f"Samples evaluated: {len(references)}\n")
                tf.write(f"Errors: {errors}\n\n")
                tf.write("AGGREGATE METRICS\n")
                tf.write("-" * 80 + "\n")
                for metric, vals in agg_now.items():
                    tf.write(f"  {metric:12s} mean={vals['mean']:.4f}  std={vals['std']:.4f}\n")
                tf.write("\n" + "=" * 80 + "\n\n")
                for ps2 in per_sample:
                    tf.write("-" * 80 + "\n")
                    tf.write(f"STUDY ID: {ps2['study_id']}\n")
                    tf.write(f"IMAGE: {ps2['image_path']}\n")
                    tf.write(f"PREDICTED FINDINGS: {ps2['predicted_findings']}\n")
                    tf.write(f"REFERENCE FINDINGS: {ps2['reference_findings']}\n")
                    tf.write("-" * 80 + "\n")
                    tf.write("GENERATED REPORT:\n")
                    tf.write("-" * 80 + "\n")
                    tf.write((ps2.get("generated_report") or "").strip() + "\n")
                    tf.write("-" * 80 + "\n")
                    tf.write("ORIGINAL REPORT:\n")
                    tf.write("-" * 80 + "\n")
                    tf.write((ps2.get("reference_report") or "").strip() + "\n")
                    tf.write("\n")

            print(f"  [{idx + 1}/{len(items)}] {study_id} complete — {len(references)} total done")

        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  ERROR on {study_id}: {exc}")
            continue

    # ---------- Aggregate metrics (final) ----------
    agg = evaluate_reports(references, candidates) if references else {}

    # Final save (ensures last state is written; incremental saves already persisted per sample).
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "IU X-Ray (Demner-Fushman et al., 2015)",
            "test_split": f"{split} split (7:2:1 split, seed=42)",
            "num_samples": len(references),
            "num_errors": errors,
            "classifier": {
                "model_name": engine.model_name,
                "architecture": model_info.architecture,
                "parameter_count": model_info.parameter_count,
                "authors": model_info.authors,
                "organization": model_info.organization,
                "publication": model_info.publication,
                "repository": model_info.repository,
                "training_datasets": model_info.training_datasets,
                "training_objective": model_info.training_objective,
                "loss_function": model_info.loss_function,
                "input_dimensions": model_info.input_dimensions,
                "preprocessing": model_info.preprocessing,
                "multi_label_head": model_info.multi_label_head,
                "supported_labels": [
                    {"requested_label": ls.requested_label, "supported": ls.supported,
                     "model_output_name": ls.model_output_name}
                    for ls in model_info.supported_labels
                ],
                "license": model_info.license,
                "permitted_uses": model_info.permitted_uses,
                "limitations": model_info.limitations,
                "evaluation_datasets": model_info.evaluation_datasets,
                "published_metrics": model_info.published_metrics,
                "weights_url": model_info.weights_url,
                "trained_by_this_project": model_info.trained_by_this_project,
            },
            "llm": {
                "model_id": effective_model_id,
                "base_model_id": llm_model_id,
                "finetuned": finetuned_model_dir is not None,
                "backend": "transformers_local",
                "device": device,
                "configured": llm_client.is_configured(),
            },
            "metrics": agg,
            "per_sample": per_sample,
        }, f, indent=2)

    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w") as tf:
        tf.write("=" * 80 + "\n")
        tf.write("CHEST X-PERT — V7 GENERATED vs ORIGINAL REPORT COMPARISON\n")
        tf.write("=" * 80 + "\n\n")
        tf.write(f"Split: {split}\n")
        tf.write(f"Samples evaluated: {len(references)}\n")
        tf.write(f"Errors: {errors}\n\n")
        tf.write("AGGREGATE METRICS\n")
        tf.write("-" * 80 + "\n")
        for metric, vals in agg.items():
            tf.write(f"  {metric:12s} mean={vals['mean']:.4f}  std={vals['std']:.4f}\n")
        tf.write("\n" + "=" * 80 + "\n\n")

        for ps in per_sample:
            tf.write("-" * 80 + "\n")
            tf.write(f"STUDY ID: {ps['study_id']}\n")
            tf.write(f"IMAGE: {ps['image_path']}\n")
            tf.write(f"PREDICTED FINDINGS: {ps['predicted_findings']}\n")
            tf.write(f"REFERENCE FINDINGS: {ps['reference_findings']}\n")
            tf.write("-" * 80 + "\n")
            tf.write("GENERATED REPORT:\n")
            tf.write("-" * 80 + "\n")
            tf.write((ps.get("generated_report") or "").strip() + "\n")
            tf.write("-" * 80 + "\n")
            tf.write("ORIGINAL REPORT:\n")
            tf.write("-" * 80 + "\n")
            tf.write((ps.get("reference_report") or "").strip() + "\n")
            tf.write("\n")

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Samples evaluated: {len(references)}")
    print(f"Errors: {errors}")
    print()
    print("Aggregate metrics:")
    for metric, vals in agg.items():
        print(f"  {metric:12s} mean={vals['mean']:.4f}  std={vals['std']:.4f}")
    print()
    print(f"Metrics JSON saved to: {out_path}")
    print(f"Text comparison saved to: {txt_path}")

    return overall


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on IU X-Ray")
    parser.add_argument("--data-root", default="./data/iu_xray")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit test samples (for quick tests)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate (train, val, test)")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--finetuned-model", default=None,
                        help="Path to fine-tuned model directory (overrides --llm-model)")
    parser.add_argument("--llm-api-url", default=None)
    parser.add_argument("--llm-api-token", default=None)
    parser.add_argument("--output", default="./checkpoints/iu_xray_evaluation_metrics.json")
    args = parser.parse_args()

    evaluate_iu_xray(
        data_root=args.data_root,
        max_samples=args.max_samples,
        device=args.device,
        split=args.split,
        llm_model_id=args.llm_model,
        finetuned_model_dir=args.finetuned_model,
        llm_api_url=args.llm_api_url,
        llm_api_token=args.llm_api_token,
        output_file=args.output,
    )
