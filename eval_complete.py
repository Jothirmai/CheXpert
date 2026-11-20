"""Comprehensive evaluation: CNN classification metrics + LLM report metrics
on IU X-Ray test set. All results in one table + saved to JSON.
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

from chest_xpert.config import MODEL_CONFIG, LABELS_18
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
from cliq_rrg.medical_rag import build_rag_index_from_samples


def extract_findings(report_text):
    tl = report_text.lower()
    return [f for f in IU_FINDINGS if f.lower() in tl]


def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_v7_prompt(disease_probs, patient_id, view, rag_context=""):
    lines = [
        "Generate a V7 Qwen3-VL All-Disease Radiologist Evidence Fusion report.",
        "",
        "AI classifier disease probabilities:",
    ]
    for d in sorted(disease_probs, key=lambda x: -disease_probs[x]):
        lines.append("- %s: %.4f" % (d, disease_probs[d]))
    lines.append("")
    lines.append("Patient ID: %s" % patient_id)
    lines.append("View: %s" % (view or "Posteroanterior (PA)"))
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
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-model", default="./checkpoints/llm_v7_rag_finetuned")
    parser.add_argument("--output", default="./checkpoints/complete_evaluation_metrics.json")
    parser.add_argument("--k-retrieve", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Device: %s" % device, flush=True)

    downloader = IUXRayDownloader(root=args.data_root)
    downloader.download_all(force=False)
    reports = parse_reports(downloader.reports_dir)
    samples = build_samples(downloader.images_dir, reports)
    train_s, val_s, test_s = train_val_test_split(samples, seed=42)
    if args.max_samples:
        test_s = test_s[:args.max_samples]
    print("Test samples: %d" % len(test_s), flush=True)

    rag = build_rag_index_from_samples(train_s)

    engine = InferenceEngine(device=str(device))
    _ = engine.model
    print("Classifier loaded", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype="auto").to(device)
    model.eval()
    print("LLM loaded", flush=True)

    y_true = {d: [] for d in LABELS_18}
    y_pred = {d: [] for d in LABELS_18}

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

            ref_findings = extract_findings(reference_report)
            for d in LABELS_18:
                y_true[d].append(1 if d in ref_findings else 0)
                y_pred[d].append(1 if disease_probs.get(d, 0) > 0.5 else 0)

            pred_diseases = [d for d, p in disease_probs.items() if p > 0.5]
            retrieved = rag.retrieve(pred_diseases, k=args.k_retrieve)
            retrieved = [r for r in retrieved if r["study_id"] != study_id][:args.k_retrieve]
            rag_lines = []
            if retrieved:
                rag_lines.append("Retrieved similar clinical cases for reference:")
                for r in retrieved:
                    rag_lines.append("- Case %s (shared: %d): %s" % (r["study_id"], r["overlap"], r["report_text"][:250]))
                rag_lines.append("")
            rag_context = "\n".join(rag_lines)

            prompt = build_v7_prompt(disease_probs, study_id, item.view, rag_context)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(out[0], skip_special_tokens=True)
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()

            t1 = time.perf_counter()

            b = bleu(reference_report, generated)
            m_ = meteor(reference_report, generated)
            r_ = rouge_l(reference_report, generated)
            c = cider([reference_report], generated)
            bs = bert_score(reference_report, generated)

            references.append(reference_report)
            candidates.append(generated)

            per_sample.append({
                "study_id": study_id,
                "bleu": b, "meteor": m_, "rouge_l": r_, "cider": c, "bert_score": bs,
                "cnn_diseases": disease_probs,
                "ref_findings": ref_findings,
                "generated_report": generated,
                "reference_report": reference_report,
                "latency_s": t1 - t0,
            })

            print("  [%d/%d] %s: BLEU=%.3f METEOR=%.3f ROUGE-L=%.3f BERT=%.3f (%.1fs)" % (
                i + 1, len(test_s), study_id, b, m_, r_, bs, t1 - t0), flush=True)

        except Exception as exc:
            errors += 1
            print("  ERROR %s: %s" % (study_id, exc), flush=True)
            continue

    print()
    print("=" * 85)
    print("CNN CLASSIFICATION METRICS (threshold=0.5)")
    print("=" * 85)
    print("%-28s %7s %7s %7s %9s" % ("Disease", "Prec", "Rec", "F1", "Support"))
    print("-" * 85)

    per_label = {}
    for d in LABELS_18:
        p, r, f = precision_recall_f1(y_true[d], y_pred[d])
        support = sum(y_true[d])
        per_label[d] = {"precision": p, "recall": r, "f1": f, "support": support}
        print("%-28s %7.3f %7.3f %7.3f %9d" % (d, p, r, f, support))

    print("-" * 85)
    macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
    macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)

    tp_all = sum(1 for d in LABELS_18 for t, p in zip(y_true[d], y_pred[d]) if t == 1 and p == 1)
    fp_all = sum(1 for d in LABELS_18 for t, p in zip(y_true[d], y_pred[d]) if t == 0 and p == 1)
    fn_all = sum(1 for d in LABELS_18 for t, p in zip(y_true[d], y_pred[d]) if t == 1 and p == 0)
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print("%-28s %7.3f %7.3f %7.3f %9s" % ("MACRO AVG", macro_p, macro_r, macro_f1, ""))
    print("%-28s %7.3f %7.3f %7.3f %9s" % ("MICRO AVG", micro_p, micro_r, micro_f1, ""))
    print("=" * 85)

    agg = evaluate_reports(references, candidates) if references else {}

    print()
    print("=" * 85)
    print("LLM V7 REPORT GENERATION METRICS")
    print("=" * 85)
    for k, v in agg.items():
        print("  %-12s: mean=%.4f  std=%.4f" % (k, v["mean"], v["std"]))
    print("=" * 85)

    print()
    print("=" * 85)
    print("COMBINED RESULTS TABLE")
    print("=" * 85)
    print("%-30s %-20s %12s" % ("Component", "Metric", "Value"))
    print("-" * 85)
    print("%-30s %-20s %12.4f" % ("CNN (18 diseases)", "Macro Precision", macro_p))
    print("%-30s %-20s %12.4f" % ("", "Macro Recall", macro_r))
    print("%-30s %-20s %12.4f" % ("", "Macro F1", macro_f1))
    print("%-30s %-20s %12.4f" % ("", "Micro Precision", micro_p))
    print("%-30s %-20s %12.4f" % ("", "Micro Recall", micro_r))
    print("%-30s %-20s %12.4f" % ("", "Micro F1", micro_f1))
    print("%-30s %-20s %12.4f" % ("LLM (V7 report)", "BLEU", agg.get("bleu", {}).get("mean", 0)))
    print("%-30s %-20s %12.4f" % ("", "METEOR", agg.get("meteor", {}).get("mean", 0)))
    print("%-30s %-20s %12.4f" % ("", "ROUGE-L", agg.get("rouge_l", {}).get("mean", 0)))
    print("%-30s %-20s %12.4f" % ("", "CIDEr", agg.get("cider", {}).get("mean", 0)))
    print("%-30s %-20s %12.4f" % ("", "BERTScore", agg.get("bert_score", {}).get("mean", 0)))
    print("=" * 85)

    from chest_xpert.model_registry import registry
    model_info = registry()[MODEL_CONFIG.name]

    overall = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "IU X-Ray (Demner-Fushman et al., 2015)",
        "test_split": "10% (7:2:1 split, seed=42)",
        "num_samples": len(references),
        "num_errors": errors,
        "cnn_metrics": {
            "per_label": per_label,
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        },
        "llm_metrics": agg,
        "classifier": {
            "model_name": engine.model_name,
            "architecture": model_info.architecture,
            "parameter_count": model_info.parameter_count,
            "training_datasets": model_info.training_datasets,
            "weights_url": model_info.weights_url,
            "trained_by_this_project": model_info.trained_by_this_project,
            "input_dimensions": model_info.input_dimensions,
        },
        "llm": {
            "model_id": args.llm_model,
            "backend": "transformers_local",
            "device": str(device),
            "rag_enabled": True,
            "k_retrieve": args.k_retrieve,
        },
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)

    print("\nAll metrics saved to: %s" % out_path)


if __name__ == "__main__":
    main()
