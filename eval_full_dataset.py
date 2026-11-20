"""Comprehensive evaluation on ALL IU X-Ray images (train+val+test).
Fixed CNN thresholding (sigmoid + per-disease threshold from validation).
Saves results to new JSON file.
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


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x)).item() if isinstance(x, torch.Tensor) else 1.0 / (1.0 + __import__("math").exp(-x))


def find_best_threshold(y_true, scores):
    """Find threshold that maximizes F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in [i / 100 for i in range(5, 95)]:
        y_pred = [1 if s > t else 0 for s in scores]
        tp = sum(1 for t_, p in zip(y_true, y_pred) if t_ == 1 and p == 1)
        fp = sum(1 for t_, p in zip(y_true, y_pred) if t_ == 0 and p == 1)
        fn = sum(1 for t_, p in zip(y_true, y_pred) if t_ == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/iu_xray")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-model", default="./checkpoints/llm_v7_rag_finetuned")
    parser.add_argument("--output", default="./checkpoints/full_dataset_evaluation.json")
    parser.add_argument("--llm-max-samples", type=int, default=None,
                        help="Limit LLM samples (None=all, slow! Use 0 to skip LLM)")
    parser.add_argument("--k-retrieve", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    print("Device: %s" % device, flush=True)
    print("=" * 70)
    print("FULL DATASET EVALUATION (train+val+test)")
    print("=" * 70, flush=True)

    # ---------- Load ALL data ----------
    downloader = IUXRayDownloader(root=args.data_root)
    downloader.download_all(force=False)
    reports = parse_reports(downloader.reports_dir)
    samples = build_samples(downloader.images_dir, reports)
    print("Total samples: %d" % len(samples), flush=True)

    train_s, val_s, test_s = train_val_test_split(samples, seed=42)
    print("Train=%d Val=%d Test=%d" % (len(train_s), len(val_s), len(test_s)), flush=True)

    # ---------- Build RAG index from TRAIN only ----------
    rag = build_rag_index_from_samples(train_s)

    # ---------- Load classifier ----------
    engine = InferenceEngine(device=str(device))
    _ = engine.model
    print("Classifier loaded", flush=True)

    # ========================================================================
    # PHASE 1: CNN on ALL images, tune thresholds on VAL
    # ========================================================================
    print()
    print("=" * 70)
    print("PHASE 1: CNN classification on ALL images")
    print("=" * 70, flush=True)

    # Get scores for val set to tune thresholds
    print("Scoring val set for threshold tuning...", flush=True)
    val_scores = {d: [] for d in LABELS_18}
    val_y_true = {d: [] for d in LABELS_18}
    for item in val_s:
        pp = preprocess(item.image_path, str(device), resolution=MODEL_CONFIG.input_resolution)
        result = engine.predict(pp.tensor)
        mappings = map_outputs_to_labels(result.raw, engine.model_name)
        probs = {m.requested_label: (m.score if m.score is not None else 0.0) for m in mappings if m.supported}
        ref_findings = extract_findings(item.report_text.strip())
        for d in LABELS_18:
            val_scores[d].append(probs.get(d, 0.0))
            val_y_true[d].append(1 if d in ref_findings else 0)

    # Tune thresholds
    thresholds = {}
    for d in LABELS_18:
        t, f1 = find_best_threshold(val_y_true[d], val_scores[d])
        thresholds[d] = t
    print("Thresholds tuned on val set", flush=True)

    # Run CNN on ALL images
    print("Running CNN on all %d images..." % len(samples), flush=True)
    all_cnn_results = {}  # study_id -> disease_probs
    all_ref_findings = {}  # study_id -> findings
    all_report_texts = {}

    t_start = time.time()
    for idx, item in enumerate(samples):
        try:
            pp = preprocess(item.image_path, str(device), resolution=MODEL_CONFIG.input_resolution)
            result = engine.predict(pp.tensor)
            mappings = map_outputs_to_labels(result.raw, engine.model_name)
            probs = {m.requested_label: (m.score if m.score is not None else 0.0) for m in mappings if m.supported}
            all_cnn_results[item.study_id] = probs
            all_ref_findings[item.study_id] = extract_findings(item.report_text.strip())
            all_report_texts[item.study_id] = item.report_text.strip()
        except Exception as exc:
            print("  CNN ERROR %s: %s" % (item.study_id, exc), flush=True)
            continue
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print("  [%d/%d] %.1fs elapsed" % (idx + 1, len(samples), elapsed), flush=True)

    elapsed = time.time() - t_start
    print("CNN done: %d images in %.1fs" % (len(all_cnn_results), elapsed), flush=True)

    # CNN metrics on ALL
    print()
    print("-" * 70)
    print("CNN PER-DISEASE METRICS (ALL images, tuned thresholds)")
    print("-" * 70)
    print("%-28s %8s %8s %8s %9s" % ("Disease", "Prec", "Rec", "F1", "Support"))
    print("-" * 70)

    per_label = {}
    for d in LABELS_18:
        y_t = [all_ref_findings[sid].count(d) > 0 for sid in all_cnn_results if sid in all_ref_findings]
        y_p = [1 if all_cnn_results[sid].get(d, 0) > thresholds[d] else 0 for sid in all_cnn_results if sid in all_ref_findings]
        # Actually need proper alignment
        pass

    # Simpler: iterate all images
    y_true_all = {d: [] for d in LABELS_18}
    y_pred_all = {d: [] for d in LABELS_18}
    for sid in all_cnn_results:
        if sid not in all_ref_findings:
            continue
        rf = all_ref_findings[sid]
        probs = all_cnn_results[sid]
        for d in LABELS_18:
            y_true_all[d].append(1 if d in rf else 0)
            y_pred_all[d].append(1 if probs.get(d, 0) > thresholds[d] else 0)

    macro_p = macro_r = macro_f1 = 0.0
    per_label = {}
    for d in LABELS_18:
        p, r, f = precision_recall_f1(y_true_all[d], y_pred_all[d])
        support = sum(y_true_all[d])
        per_label[d] = {"precision": p, "recall": r, "f1": f, "support": support, "threshold": thresholds[d]}
        macro_p += p; macro_r += r; macro_f1 += f
        print("%-28s %8.4f %8.4f %8.4f %9d (thr=%.3f)" % (d, p, r, f, support, thresholds[d]))
    n_labels = len([d for d in LABELS_18 if sum(y_true_all[d]) > 0])
    macro_p = sum(v["precision"] for v in per_label.values() if v["support"] > 0) / max(n_labels, 1)
    macro_r = sum(v["recall"] for v in per_label.values() if v["support"] > 0) / max(n_labels, 1)
    macro_f1 = sum(v["f1"] for v in per_label.values() if v["support"] > 0) / max(n_labels, 1)
    print("-" * 70)
    print("%-28s %8.4f %8.4f %8.4f" % ("MACRO AVG", macro_p, macro_r, macro_f1))
    print("=" * 70)

    # ========================================================================
    # PHASE 2: LLM V7 report on images (with RAG)
    # ========================================================================
    if args.llm_max_samples == 0:
        print()
        print("=" * 70)
        print("PHASE 2: SKIPPED (llm-max-samples=0)")
        print("=" * 70, flush=True)
        per_sample = []
        references = []
        candidates = []
        agg = {}
        llm_errors = 0
    else:
        print()
        print("=" * 70)
        print("PHASE 2: LLM V7 report generation")
        print("=" * 70, flush=True)

    # Decide which images to run LLM on
    llm_samples = samples if args.llm_max_samples is None else samples[:args.llm_max_samples]
    print("Running LLM on %d images..." % len(llm_samples), flush=True)
    print("(Each ~27s on GPU; est. %.1f min total)" % (len(llm_samples) * 27 / 60), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype="auto").to(device)
    model.eval()
    print("LLM loaded", flush=True)

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

    references = []
    candidates = []
    per_sample = []
    llm_errors = 0
    t_start = time.time()

    for i, item in enumerate(llm_samples):
        study_id = item.study_id
        reference_report = item.report_text.strip()
        if len(reference_report) < 30:
            llm_errors += 1
            continue

        try:
            t0 = time.perf_counter()
            probs = all_cnn_results.get(study_id, {})
            pred_diseases = [d for d, p in probs.items() if p > thresholds.get(d, 0.5)]
            retrieved = rag.retrieve(pred_diseases, k=args.k_retrieve)
            retrieved = [r for r in retrieved if r["study_id"] != study_id][:args.k_retrieve]
            rag_lines = []
            if retrieved:
                rag_lines.append("Retrieved similar clinical cases for reference:")
                for r in retrieved:
                    rag_lines.append("- Case %s (shared: %d): %s" % (r["study_id"], r["overlap"], r["report_text"][:250]))
                rag_lines.append("")
            rag_context = "\n".join(rag_lines)

            prompt = build_v7_prompt(probs, study_id, item.view, rag_context)
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
                "cnn_probs": probs,
                "ref_findings": extract_findings(reference_report),
                "generated_report": generated,
                "reference_report": reference_report,
                "latency_s": t1 - t0,
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print("  [%d/%d] %.1fs elapsed, avg metrics so far..." % (i + 1, len(llm_samples), elapsed), flush=True)

        except Exception as exc:
            llm_errors += 1
            print("  LLM ERROR %s: %s" % (study_id, exc), flush=True)
            continue

    elapsed = time.time() - t_start
    print("LLM done: %d images in %.1fs (%.1fs each)" % (len(per_sample), elapsed, elapsed / max(len(per_sample), 1)), flush=True)

    # ---------- Aggregate LLM metrics ----------
    agg = evaluate_reports(references, candidates) if references else {}

    print()
    print("=" * 70)
    print("FINAL RESULTS (ALL DATASET)")
    print("=" * 70)
    print("CNN Macro F1: %.4f" % macro_f1)
    print("CNN Macro Precision: %.4f" % macro_p)
    print("CNN Macro Recall: %.4f" % macro_r)
    print()
    print("LLM Metrics:")
    for k, v in agg.items():
        print("  %-12s: mean=%.4f  std=%.4f" % (k, v["mean"], v["std"]))
    print("=" * 70)

    # ---------- Save ----------
    from chest_xpert.model_registry import registry
    model_info = registry()[MODEL_CONFIG.name]

    overall = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "IU X-Ray (Demner-Fushman et al., 2015)",
        "split": "train+val+test (all %d images)" % len(samples),
        "cnn_images_evaluated": len(all_cnn_results),
        "llm_images_evaluated": len(per_sample),
        "llm_errors": llm_errors,
        "cnn_metrics": {
            "per_label": per_label,
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "thresholds": thresholds,
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

    print("\nAll results saved to: %s" % out_path)


if __name__ == "__main__":
    main()
