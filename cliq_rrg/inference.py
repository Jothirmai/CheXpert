"""End-to-end CliQ-RRG inference: Stage 1 (CNN alignment) + Stage 2 (GraphRAG +
LLM V7-format radiology report generation).

Usage:
    python cliq_rrg/inference.py --image samples/sample_frontal.png \
        --cnn-model checkpoints/cnn_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliq_rrg.stage1 import Stage1CNN
from cliq_rrg.stage2 import (
    ClinicalKnowledgeGraph, GraphRAGRetriever, V7LLMClient,
    PatientInfo, AIEvidence, build_kg_from_reports,
)
from cliq_rrg import CNN_MODEL_PATH
from cliq_rrg.dataset import IU_FINDINGS


def load_cnn_model(ckpt_path: str | Path, device: str) -> Stage1CNN:
    path = str(ckpt_path)
    if not Path(path).exists():
        print(f"Warning: checkpoint not found at {path}; using untrained weights.")
        return Stage1CNN(num_diseases=len(IU_FINDINGS), pretrained=False).to(device)
    return Stage1CNN.load(
        path, num_diseases=len(IU_FINDINGS), pretrained=False, device=device
    )


def classify_image(model: Stage1CNN, image_path: str, device: str,
                   resolution: int = 224):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]

    model.eval()
    with torch.no_grad():
        out = model(img)
    probs = out["disease_probs"][0].cpu().numpy()
    logits = out["disease_logits"][0].cpu().numpy()

    results = {}
    for i, disease in enumerate(IU_FINDINGS):
        results[disease] = {
            "probability": float(probs[i]),
            "logit": float(logits[i]),
            "positive": bool(probs[i] > 0.5),
        }
    return results, out


def build_knowledge_graph(data_root: str | Path = "./data/iu_xray") -> ClinicalKnowledgeGraph:
    """Build KG from IU X-Ray training reports (or empty if data unavailable)."""
    kg = ClinicalKnowledgeGraph()
    try:
        from cliq_rrg.dataset import IUXRayDownloader, parse_reports
        downloader = IUXRayDownloader(root=data_root)
        reports = parse_reports(downloader.reports_dir)
        for rid, text in list(reports.items())[:500]:  # limit for speed
            kg.add_report(rid, text)
    except Exception as e:
        print(f"Warning: could not build KG from dataset ({e}); using empty KG.")
    return kg


def run_inference(
    image_path: str,
    cnn_model_path: str | Path = CNN_MODEL_PATH,
    device: str | None = None,
    data_root: str | Path = "./data/iu_xray",
    llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    llm_api_url: str | None = None,
    llm_api_token: str | None = None,
    load_in_4bit: bool = True,
    lora_adapter_path: str | None = None,
    patient_id: str = "NOT PROVIDED",
    view: str = "Posteroanterior (PA)",
):
    import torch

    # Resolve device: pick CUDA GPU with most free memory (Tesla V100 32GB).
    if device is None:
        if torch.cuda.is_available():
            best_idx = 0
            best_free = 0
            for i in range(torch.cuda.device_count()):
                try:
                    free, _ = torch.cuda.mem_get_info(i)
                    if free > best_free:
                        best_free = free
                        best_idx = i
                except Exception:
                    pass
            torch.cuda.set_device(best_idx)
            device = f"cuda:{best_idx}"
            print(f"Device: {device} ({best_free / 1024**3:.1f} GB free on {torch.cuda.get_device_name(best_idx)})")
        else:
            device = "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")
    print(f"Image: {image_path}")

    # ---------- Stage 1: CNN classification ----------
    print("\n[Stage 1] Disease-aware Visual-Textual Alignment...")
    model = load_cnn_model(cnn_model_path, device)
    disease_results, stage1_out = classify_image(model, image_path, device)

    positive_diseases = [d for d, r in disease_results.items() if r["positive"]]
    uncertain_diseases = [d for d, r in disease_results.items()
                          if 0.2 < r["probability"] <= 0.5]
    all_relevant = positive_diseases + uncertain_diseases

    print(f"  Positive findings: {positive_diseases if positive_diseases else 'none'}")
    print(f"  Uncertain findings: {uncertain_diseases if uncertain_diseases else 'none'}")

    # ---------- Stage 2: GraphRAG retrieval + LLM V7 report ----------
    print("\n[Stage 2] GraphRAG retrieval + LLM V7 report generation...")
    kg = build_knowledge_graph(data_root)
    retriever = GraphRAGRetriever(kg)
    evidence = retriever.retrieve_evidence(
        all_relevant if all_relevant else IU_FINDINGS[:5]
    )

    client = V7LLMClient(
        model_id=llm_model_id,
        device=device,
        api_url=llm_api_url,
        api_token=llm_api_token,
        load_in_4bit=load_in_4bit,
        lora_adapter_path=lora_adapter_path,
    )

    disease_probs = {d: disease_results[d]["probability"] for d in all_relevant}

    # Build AI evidence items (with placeholder gradcam locations)
    ai_evidence_items = []
    for d in all_relevant:
        p = disease_results[d]["probability"]
        ai_evidence_items.append(AIEvidence(
            disease=d,
            probability=p,
            gradcam_location="Grad-CAM attention map (not computed in CLI mode)",
            supported=p > 0.5,
            notes=(
                "Supported by visible imaging findings" if p > 0.5 else
                "Probability below 0.5 threshold"
            ),
        ))

    patient = PatientInfo(
        patient_id=patient_id,
        study="Chest X-ray",
        view=view,
        patient_age="NOT PROVIDED",
        patient_sex="NOT PROVIDED",
        clinical_indication="NOT PROVIDED",
        comparison_study="NOT PROVIDED",
    )

    v7_report = client.generate_v7_report(
        patient=patient,
        diseases=all_relevant,
        disease_probs=disease_probs,
        evidence=evidence,
        ai_evidence_items=ai_evidence_items,
        image_summary="Frontal chest X-ray provided for analysis.",
        image_path=image_path,
    )

    # ---------- Output ----------
    print("\n" + "=" * 80)
    print("CLIQ-RRG V7 RADIOLOGY REPORT")
    print("=" * 80)
    print(v7_report.full_text)

    return {
        "stage1": disease_results,
        "stage2": {
            "v7_report": v7_report.full_text,
            "patient": v7_report.patient,
            "exam_technique": v7_report.exam_technique,
            "image_quality": v7_report.image_quality,
            "findings": v7_report.findings,
            "ai_evidence": v7_report.ai_evidence,
            "impression": v7_report.impression,
            "diseases": v7_report.diseases,
            "confidence": v7_report.confidence,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CliQ-RRG V7 inference")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--cnn-model", default=CNN_MODEL_PATH,
                        help="Path to cnn_model.pth checkpoint")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--data-root", default="./data/iu_xray")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-api-url", default=None)
    parser.add_argument("--llm-api-token", default=None)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        default=True, help="Disable 4-bit quantization")
    parser.add_argument("--lora-adapter", dest="lora_adapter_path", default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--patient-id", default="NOT PROVIDED")
    parser.add_argument("--view", default="Posteroanterior (PA)")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        cnn_model_path=args.cnn_model,
        device=args.device,
        data_root=args.data_root,
        llm_model_id=args.llm_model,
        llm_api_url=args.llm_api_url,
        llm_api_token=args.llm_api_token,
        patient_id=args.patient_id,
        view=args.view,
    )


if __name__ == "__main__":
    main()
