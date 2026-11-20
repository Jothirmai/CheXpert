"""Fine-tune Qwen/Qwen2.5-VL-7B-Instruct on IU X-Ray radiology reports with
Medical RAG (Retrieval-Augmented Generation) using LoRA (PEFT).

LoRA adds small trainable adapter matrices to the frozen base model, making
fine-tuning fast and memory-efficient (~1-2 GB VRAM for 0.5B model). At
inference time, adapters are applied on the fly so output behaviour is
preserved for inputs outside the fine-tuning distribution while task-specific
responses improve and latency stays low.

At training time, for each sample we retrieve k similar reports (by shared
disease labels) from the training set and prepend them as clinical evidence
context. This teaches the LLM to ground its V7 report in retrieved evidence.

At inference time (cliq_rrg/inference.py), the same RAG retrieval is used
and the LoRA adapter is loaded via --lora-adapter.

Usage:
    source /home/doli/anaconda3/bin/activate hct-lsr
    pip install peft bitsandbytes accelerate
    python cliq_rrg/finetune_llm.py \
        --data-root ./data/iu_xray \
        --output-dir ./checkpoints/llm_v7_lora \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --device cuda \
        --epochs 2 --batch-size 4 --lr 1e-4 \
        --lora-r 16 --lora-alpha 32 --load-in-4bit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliq_rrg.dataset import (
    IUXRayDownloader, parse_reports, build_samples, train_val_test_split,
    IU_FINDINGS,
)
from cliq_rrg.medical_rag import MedicalRAGIndex, build_rag_index_from_samples


def extract_findings(report_text: str) -> list[str]:
    tl = report_text.lower()
    return [f for f in IU_FINDINGS if f.lower() in tl]


class V7FineTuneDataset(Dataset):
    """(prompt_with_rag, target) pairs for fine-tuning V7 report generation."""

    def __init__(
        self,
        samples,
        rag_index: MedicalRAGIndex,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        k_retrieve: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        count = 0
        for s in samples:
            if max_samples and count >= max_samples:
                break
            report_text = s.report_text.strip()
            if len(report_text) < 30:
                continue

            findings = extract_findings(report_text)

            # Retrieve similar reports (exclude self)
            retrieved = rag_index.retrieve(findings, k=k_retrieve + 1)
            retrieved = [r for r in retrieved if r["study_id"] != s.study_id][:k_retrieve]

            # Build RAG context
            rag_lines = []
            if retrieved:
                rag_lines.append("Retrieved similar clinical cases for reference:")
                for r in retrieved:
                    snippet = r["report_text"][:300]
                    rag_lines.append(
                        f"- Case {r['study_id']} (shared: {r['overlap']}): {snippet}"
                    )
                rag_lines.append("")
            rag_context = "\n".join(rag_lines)

            # Build conditioning prompt
            prompt_lines = [
                "Generate a V7 Qwen3-VL All-Disease Radiologist Evidence Fusion report.",
                "",
                "AI classifier disease probabilities:",
            ]
            for d in findings:
                prompt_lines.append(f"- {d}: 0.7500")
            if not findings:
                prompt_lines.append("- No significant findings: 0.1000")
            prompt_lines.append("")
            prompt_lines.append(f"Patient ID: {s.study_id}")
            prompt_lines.append(f"View: {s.view or 'Posteroanterior (PA)'}")
            prompt_lines.append("")
            prompt_lines.append(rag_context)
            prompt_lines.append("Report:")

            prompt = "\n".join(prompt_lines)
            target = report_text.strip()

            self.examples.append((prompt, target))
            count += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, target = self.examples[idx]
        full_text = prompt + "\n" + target

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding="max_length",
        )

        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_enc["input_ids"])

        labels = enc["input_ids"][0].clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
        }


def apply_lora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
               target_modules: Optional[list] = None):
    """Apply LoRA adapters to the model using PEFT."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if target_modules is not None and len(target_modules) > 0:
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def finetune(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- Load dataset ----------
    downloader = IUXRayDownloader(root=args.data_root)
    downloader.download_all(force=False)

    reports = parse_reports(downloader.reports_dir)
    samples = build_samples(downloader.images_dir, reports)
    train_s, val_s, _ = train_val_test_split(samples, seed=42)

    if args.max_train_samples:
        train_s = train_s[: args.max_train_samples]

    print(f"Train samples: {len(train_s)}")

    # ---------- Build RAG index from training reports ----------
    rag = build_rag_index_from_samples(train_s)

    # ---------- Load model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pick GPU with most free memory (Tesla V100 32GB).
    load_device = str(device)
    if device.type == "cuda" and torch.cuda.is_available():
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
        load_device = f"cuda:{best_idx}"
        print(f"Selected GPU {best_idx} ({best_free / 1024**3:.1f} GB free)")

    load_kwargs: dict = {}
    if args.load_in_4bit and device.type == "cuda":
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": load_device}
        except Exception:
            load_kwargs["torch_dtype"] = "auto"
            load_kwargs["device_map"] = {"": load_device}
    else:
        load_kwargs["torch_dtype"] = "auto"
        load_kwargs["device_map"] = {"": load_device}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=True,
        trust_remote_code=True,
        **load_kwargs,
    )
    model_device = next(model.parameters()).device
    print(f"Base model '{args.model_id}' loaded on device: {model_device}")

    # ---------- Apply LoRA ----------
    model = apply_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    model.train()

    train_ds = V7FineTuneDataset(
        train_s, rag, tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
        k_retrieve=args.k_retrieve,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Only LoRA parameters are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.4f}")

    # ---------- Save LoRA adapter only (not full model) ----------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune LLM for V7 report generation with Medical RAG"
    )
    parser.add_argument("--data-root", default="./data/iu_xray",
                        help="Root directory for IU X-Ray dataset")
    parser.add_argument("--output-dir", default="./checkpoints/llm_v7_lora",
                        help="Directory to save LoRA adapter")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch device")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=300)
    parser.add_argument("--k-retrieve", type=int, default=3,
                        help="Number of similar reports to retrieve as context")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Target modules for LoRA")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (CUDA only)")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    raise SystemExit("This text-only trainer cannot train the active vision workflow. Run python scripts/train_report_adapter.py --help instead.")
