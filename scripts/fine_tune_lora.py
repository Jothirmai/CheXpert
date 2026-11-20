"""LEGACY trainer: do not use for the active multimodal workflow.

Use scripts/train_report_adapter.py, which preserves image-grid metadata and
uses assistant-only loss with held-out validation.

LoRA fine-tuning script for Chest Xpert Qwen VLM.

Fine-tunes the Qwen VLM (Qwen3-VL-32B or Qwen2.5-VL-7B) using LoRA
(Low-Rank Adaptation) on chest X-ray report data. The base model weights
stay frozen; only small low-rank adapter matrices are trained, making it
fast and memory-efficient (~16 GB VRAM for Qwen3-VL-32B in 4-bit on a V100).

Usage:
    export CHESTXPERT_HF_TOKEN="your_token"
    source /home/doli/anaconda3/bin/activate hct-lsr
    pip install peft bitsandbytes accelerate
    python scripts/fine_tune_lora.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --train-data /path/to/train.jsonl \
        --output-dir /home/doli/Chest Xpert/checkpoints/lora_adapter \
        --epochs 3 \
        --lr 2e-4 \
        --batch-size 1 \
        --grad-accum 16

Training data format (JSONL, one example per line):
    {"image_path": "/path/to/xray.png", "report": "V7 radiology report text..."}

Requires: peft, bitsandbytes, accelerate, transformers>=4.46, torch>=2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

# Allow running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chest_xpert.config import PROJECT_ROOT


class ChestXrayReportDataset(Dataset):
    """Dataset of (image, report) pairs for SFT fine-tuning."""

    def __init__(self, data_path: str | Path, processor, max_pixels: int = 1280 * 28 * 28):
        self.samples: list[dict] = []
        data_path = Path(data_path)
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        self.processor = processor
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        report = sample["report"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "You are a board-certified radiologist assistant. "
                            "Generate a V7 evidence-fusion radiology report "
                            "for this chest X-ray. Include sections: patient "
                            "information, exam technique, image quality, "
                            "findings (A-I), AI evidence summary, impression. "
                            "Ground every observation in the image."
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": report},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # Squeeze batch dimension.
        return {k: v.squeeze(0) for k, v in inputs.items()}


def collate_fn(batch):
    """Pad a batch of samples to the same length."""
    from transformers import DataCollatorForSeq2Seq
    # Simple approach: pad input_ids, attention_mask, pixel_values
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = input_ids.clone()
    # Mask padding tokens in labels so they don't contribute to loss.
    labels[labels == 0] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_model_and_processor(model_id: str, load_in_4bit: bool = True):
    """Load Qwen VLM in 4-bit with device_map='auto' (CUDA)."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen2VLForConditionalGeneration

    is_qwen3_vl = "Qwen3-VL" in model_id or "qwen3-vl" in model_id.lower()
    if is_qwen3_vl:
        model_class = Qwen3VLForConditionalGeneration
    else:
        model_class = Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs: dict = {}
    if load_in_4bit and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = None

    model = model_class.from_pretrained(
        model_id,
        trust_remote_code=True,
        **load_kwargs,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model_device = next(model.parameters()).device
    print(f"[fine_tune_lora] Base model '{model_id}' loaded on device: {model_device}")
    print(f"[fine_tune_lora] CUDA available: {torch.cuda.is_available()}")

    return model, processor


def apply_lora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
               target_modules: Optional[list] = None):
    """Apply LoRA adapters to the model using peft."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

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


def train(args):
    model, processor = load_model_and_processor(args.model, load_in_4bit=args.load_in_4bit)
    model = apply_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    train_dataset = ChestXrayReportDataset(args.train_data, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer: only LoRA parameters are trainable.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                print(
                    f"[fine_tune_lora] Epoch {epoch+1}/{args.epochs} "
                    f"step {global_step} loss={running_loss/args.grad_accum:.4f}"
                )
                running_loss = 0.0

        # Save checkpoint at end of each epoch.
        epoch_dir = output_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        processor.save_pretrained(str(epoch_dir))
        print(f"[fine_tune_lora] Checkpoint saved to: {epoch_dir}")

    # Save final adapter.
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"[fine_tune_lora] LoRA adapter saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen VLM for Chest Xpert")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct",
                        help="Base model id (Qwen3-VL-32B or Qwen2.5-VL-7B)")
    parser.add_argument("--train-data", required=True, help="Path to training JSONL")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "lora_adapter"),
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()

    print(f"[fine_tune_lora] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[fine_tune_lora] GPU: {torch.cuda.get_device_name(0)}")

    train(args)


if __name__ == "__main__":
    raise SystemExit("This legacy trainer is retired. Run python scripts/train_report_adapter.py --help for the corrected multimodal LoRA trainer.")
