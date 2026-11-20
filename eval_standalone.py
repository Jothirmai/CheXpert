"""Standalone evaluation: classifier + fine-tuned LLM V7 report with Medical RAG
on IU X-Ray test split.

At inference, retrieves k similar training reports (by shared disease labels)
and prepends them as clinical evidence context to the LLM prompt.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chest_xpert.config import MODEL_CONFIG
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess
from chest_xpert.label_mapping import map_outputs_to_labels
from cliq_rrg.eval import (
    bleu, meteor, rouge_l, cider, bert_score, chexbert_metrics,
    radgraph_f1, evaluate_reports,
)
from cliq_rrg.dataset import (
    IUXRayDownloader, parse_reports, build_samples, train_val_test_split,
    IU_FINDINGS,
)
from cliq_rrg.medical_rag import MedicalRAGIndex, build_rag_index_from_samples


def extract_findings(report_text: str) -> list[str]:
    tl = report_text.lower()
    return [f for f in IU_FINDINGS if f.lower() in tl]


def build_v7_prompt(disease_probs, patient_id, view, rag_context=""):
    lines = [
        "Generate a V7 Qwen3-VL All-Disease Radiologist Evidence Fusion report.",
        "",
        "AI classifier disease probabilities:",
    ]
    for d in sorted(disease_probs, key=lambda x: -disease_probs[x]):
        lines.append(f"- {d}: {disease_probs[d]:.4f}")
    lines.append("")
    lines.append(f"Patient ID: {patient_id}")
    lines.append(f"View: {view or 'Posteroanterior (PA)'}")
    lines.append("")
    if rag_context:
        lines.append(rag_context)
        lines.append("")
    lines.append("Report:")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/iu_xray")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-model", default="./checkpoints/llm_v7_rag_finetuned")
    parser.add_argument("--output", default="./checkpoints/iu_xray_evaluation_metrics_rag.json")
    parser.add_argument("--k-retrieve", type=int, default=3)
    parser.add_argument("--build-rag-from-train", action="store_true", default=True,
                        help="Build RAG index from training reports")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ---------- Load data ----------
    downloader = IUXRayDownloader(root=args.data_root)
    downloader.download_all(force=False)
    reports = parse_reports(downloader.reports_dir)
    samples = build_samples(downloader.images_dir, reports)
    train_s, val_s, test_s = train_val_test_split(samples, seed=42)
    if args.max_samples:
        test_s = test_s[: args.max_samples]
    print(f"Test samples: {len(test_s)}", flush=True)

    # ---------- Build RAG index from training reports ----------
    rag = build_rag_index_from_samples(train_s)

    # ---------- Load classifier ----------
    engine = InferenceEngine(device=str(device))
    _ = engine.model
    print("Classifier loaded", flush=True)

    # ---------- Load LLM ----------
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype="auto").to(device)
    model.eval()
    print("LLM loaded", flush=True)

    # ---------- Evaluate ----------
    references = []
    candidates = []
    per_sample = []
    errors = 0

    for i, item in enumerate(test_s):
        image_path = item.image_path
        reference_report = item.report_text.strip()
        study_id = item.study_id

        if len(reference_report) < 30:
            errors += 1
            print(f"  SKIP {study_id}: empty reference", flush=True)
            continue

        try:
            t0 = time.perf_counter()

            pp = preprocess(image_path, str(device), resolution=MODEL_CONFIG.input_resolution)
            result = engine.predict(pp.tensor)
            mappings = map_outputs_to_labels(result.raw, engine.model_name)

            disease_probs = {
                m.requested_label: (m.score if m.score is not None else 0.0)
                for m in mappings if m.supported
            }

            # ---------- RAG retrieval ----------
            pred_diseases = [d for d, p in disease_probs.items() if p > 0.5]
            retrieved = rag.retrieve(pred_diseases, k=args.k_retrieve)
            retrieved = [r for r in retrieved if r["study_id"] != study_id][:args.k_retrieve]

            rag_lines = []
            if retrieved:
                rag_lines.append("Retrieved similar clinical cases for reference:")
                for r in retrieved:
                    snippet = r["report_text"][:250]
                    rag_lines.append(
                        f"- Case {r['study_id']} (shared: {r['overlap']}): {snippet}"
                    )
                rag_lines.append("")
            rag_context = "\n".join(rag_lines)

            prompt = build_v7_prompt(disease_probs, study_id, item.view, rag_context)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out[0], skip_special_tokens=True)
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()

            t1 = time.perf_counter()

            # ---------- Metrics ----------
            b = bleu(reference_report, generated)
            m_ = meteor(reference_report, generated)
            r = rouge_l(reference_report, generated)
            c = cider([reference_report], generated)
            bs = bert_score(reference_report, generated)

            ref_findings = extract_findings(reference_report)
            pred_findings = [d for d, p in disease_probs.items() if p > 0.5]
            cx = chexbert_metrics(pred_findings, ref_findings)
            rg = radgraph_f1([(d, "present") for d in pred_findings],
                             [(d, "present") for d in ref_findings])

            references.append(reference_report)
            candidates.append(generated)

            per_sample.append({
                "study_id": study_id,
                "bleu": b, "meteor": m_, "rouge_l": r, "cider": c, "bert_score": bs,
                "chexbert": cx, "radgraph_f1": rg,
                "classifier_scores": disease_probs,
                "predicted_findings": pred_findings,
                "reference_findings": ref_findings,
                "retrieved_cases": [r["study_id"] for r in retrieved],
                "generated_report": generated,
                "reference_report": reference_report,
                "latency_s": t1 - t0,
            })

            print(
                f"  [{i+1}/{len(test_s)}] {study_id}: "
                f"BLEU={b:.4f} METEOR={m_:.4f} ROUGE-L={r:.4f} BERT={bs:.4f} "
                f"RAG={len(retrieved)} ({t1-t0:.1f}s)",
                flush=True,
            )

        except Exception as exc:
            errors += 1
            print(f"  ERROR {study_id}: {exc}", flush=True)
            continue

    # ---------- Aggregate ----------
    agg = evaluate_reports(references, candidates) if references else {}

    from chest_xpert.model_registry import registry
    model_info = registry()[MODEL_CONFIG.name]

    overall = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "IU X-Ray (Demner-Fushman et al., 2015)",
        "test_split": "10% (7:2:1 split, seed=42)",
        "num_samples": len(references),
        "num_errors": errors,
        "classifier": {
            "model_name": engine.model_name,
            "architecture": model_info.architecture,
            "parameter_count": model_info.parameter_count,
            "publication": model_info.publication,
            "repository": model_info.repository,
            "training_datasets": model_info.training_datasets,
            "loss_function": model_info.loss_function,
            "input_dimensions": model_info.input_dimensions,
            "preprocessing": model_info.preprocessing,
            "supported_labels": [
                {"requested_label": ls.requested_label, "supported": ls.supported}
                for ls in model_info.supported_labels
            ],
            "limitations": model_info.limitations,
            "weights_url": model_info.weights_url,
            "trained_by_this_project": model_info.trained_by_this_project,
        },
        "llm": {
            "model_id": args.llm_model,
            "backend": "transformers_local",
            "device": str(device),
            "rag_enabled": True,
            "k_retrieve": args.k_retrieve,
        },
        "metrics": agg,
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE (with Medical RAG)")
    print("=" * 60)
    print(f"Samples: {len(references)}, Errors: {errors}")
    print("Aggregate metrics:")
    for k, v in agg.items():
        print(f"  {k:12s}: mean={v['mean']:.4f}  std={v['std']:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
