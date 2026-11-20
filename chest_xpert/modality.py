"""Stage 0 — Chest X-ray modality validation.

Validates that an uploaded image appears to be a chest X-ray before any disease
inference runs. Uses a vision-language model (Qwen2.5-VL) as the primary method
because it can actually inspect the image, with CLIP zero-shot and a grayscale
heuristic as fallbacks.

The decision rule is a documented heuristic, NOT a validated clinical threshold.
Modality detection is fallible; the result is treated as a gate, not a
diagnosis. Technical failures (model unavailable, timeout, auth failure) are
distinguished from genuine rejections so the UI never claims an image is invalid
simply because a service failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from chest_xpert.config import MODALITY_CONFIG


class ModalityDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    TECHNICAL_FAILURE = "technical_failure"


@dataclass
class ModalityResult:
    decision: ModalityDecision
    chest_xray_prob: Optional[float] = None
    best_other_label: Optional[str] = None
    best_other_prob: Optional[float] = None
    all_probs: Optional[dict[str, float]] = None
    reason: str = ""
    model_id: str = ""
    latency_ms: float = 0.0
    technical_error: Optional[str] = None
    method: str = ""  # "qwen_vlm" | "clip_zero_shot" | "heuristic"

    @property
    def is_accepted(self) -> bool:
        return self.decision == ModalityDecision.ACCEPTED

    @property
    def is_rejected(self) -> bool:
        return self.decision == ModalityDecision.REJECTED

    @property
    def is_uncertain(self) -> bool:
        return self.decision == ModalityDecision.UNCERTAIN

    @property
    def is_technical_failure(self) -> bool:
        return self.decision == ModalityDecision.TECHNICAL_FAILURE


class ModalityValidator:
    """Lazy, cached modality validator.

    Priority order:
      1. Qwen VLM (if CHESTXPERT_HF_TOKEN is configured) — can actually see
         the image and determine if it is a chest X-ray.
      2. CLIP zero-shot — text-image similarity, works but weaker on B&W images.
      3. Grayscale-saturation heuristic — can reject color photos but cannot
         confirm an X-ray.
    """

    def __init__(self, config=None):
        self.config = config or MODALITY_CONFIG
        # Qwen VLM state
        self._qwen_model = None
        self._qwen_processor = None
        self._qwen_error: Optional[str] = None
        self._qwen_attempted: bool = False
        # CLIP state
        self._clip_model = None
        self._clip_processor = None
        self._clip_error: Optional[str] = None
        self._clip_attempted: bool = False

    # ------------------------------------------------------------------
    # Qwen VLM modality check
    # ------------------------------------------------------------------

    def _ensure_qwen_loaded(self) -> bool:
        """Load Qwen2.5-VL locally for modality validation.

        Needs GPU VRAM (~15 GB FP16 for 7B). Returns True if ready.
        """
        if self._qwen_model is not None or self._qwen_error is not None:
            return self._qwen_error is None
        self._qwen_attempted = True
        import os

        hf_token = os.environ.get("CHESTXPERT_HF_TOKEN")
        if not hf_token:
            self._qwen_error = "Qwen modality not configured: CHESTXPERT_HF_TOKEN not set."
            return False
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
            self._qwen_processor = AutoProcessor.from_pretrained(
                model_id, token=hf_token
            )
            self._qwen_model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
            )
            self._qwen_model.eval()
            qwen_device = next(self._qwen_model.parameters()).device
            print(f"[Chest Xpert] Qwen modality model loaded on device: {qwen_device}")
        except Exception as exc:  # noqa: BLE001
            self._qwen_error = f"Failed to load Qwen modality model: {exc}"
            self._qwen_model = None
            self._qwen_processor = None
            return False
        return True

    def _qwen_validate(self, image_path: str | Path, t0: float) -> ModalityResult:
        """Ask Qwen VLM to look at the image and decide if it is a chest X-ray."""
        import time
        import json as _json

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        try:
            if Path(image_path).suffix.lower() in {".dcm", ".dicom"}:
                from chest_xpert.preprocessing import load_image
                arr = load_image(image_path)
                image = Image.fromarray(np.uint8(np.clip((arr + 1024) / 2048, 0, 1) * 255)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=f"Cannot open image for Qwen modality check: {exc}",
                model_id=model_id,
                technical_error=str(exc),
                method="qwen_vlm",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if not self._ensure_qwen_loaded():
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=self._qwen_error or "Qwen model unavailable.",
                model_id=model_id,
                technical_error=self._qwen_error,
                method="qwen_vlm",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            import torch

            prompt = (
                "You are a medical imaging assistant. Look at the image and answer: "
                "Is this a chest X-ray (frontal or lateral radiograph of the chest)? "
                "Respond ONLY with a JSON object: "
                '{"is_chest_xray": true/false, "confidence": 0.0-1.0, "reason": "brief reason"}. '
                "Do not include markdown code fences."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._qwen_processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self._qwen_model.device)

            with torch.no_grad():
                out = self._qwen_model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1
                )
                generated = self._qwen_processor.decode(out[0], skip_special_tokens=True)

            # Extract JSON from response
            content = self._extract_json(generated)
            parsed = _json.loads(content)

            is_cxr = bool(parsed.get("is_chest_xray", False))
            confidence = float(parsed.get("confidence", 0.0))
            reason = str(parsed.get("reason", "No reason provided."))

            latency_ms = (time.perf_counter() - t0) * 1000.0

            if is_cxr and confidence >= 0.6:
                decision = ModalityDecision.ACCEPTED
                reason = f"Qwen VLM: {reason}"
            elif not is_cxr and confidence >= 0.6:
                decision = ModalityDecision.REJECTED
                reason = f"Qwen VLM: {reason}"
            else:
                decision = ModalityDecision.UNCERTAIN
                reason = (
                    f"Qwen VLM: insufficient confidence ({confidence:.2f}). {reason}"
                )

            return ModalityResult(
                decision=decision,
                chest_xray_prob=confidence if is_cxr else (1.0 - confidence),
                best_other_label=None,
                best_other_prob=None,
                all_probs=None,
                reason=reason,
                model_id=model_id,
                latency_ms=latency_ms,
                method="qwen_vlm",
            )
        except Exception as exc:  # noqa: BLE001
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=f"Qwen VLM inference failed: {exc}",
                model_id=model_id,
                technical_error=str(exc),
                method="qwen_vlm",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull a JSON object out of markdown/code-fence wrappers if present."""
        import re

        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # fallback: find first { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    # ------------------------------------------------------------------
    # CLIP zero-shot modality check
    # ------------------------------------------------------------------

    def _ensure_clip_loaded(self) -> bool:
        """Load CLIP on first use. Returns True if ready, False on failure."""
        if self._clip_model is not None or self._clip_error is not None:
            return self._clip_error is None
        if not self.config.enabled:
            self._clip_error = "Modality validation disabled in configuration."
            return False
        self._clip_attempted = True
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            if self.config.device == "auto":
                self.config.device = "cuda:0"
            if not self.config.device.startswith("cuda"):
                raise RuntimeError("CLIP inference requires a CUDA device")
            model_id = self.config.model_id
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import CLIPConfig
            from chest_xpert.model_revisions import CLIP_CONFIG_REVISION, CLIP_SAFETENSORS_REVISION

            self._clip_processor = CLIPProcessor.from_pretrained(
                model_id, revision=CLIP_CONFIG_REVISION
            )
            weights = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=CLIP_SAFETENSORS_REVISION,
            )
            self._clip_model = CLIPModel(
                CLIPConfig.from_pretrained(model_id, revision=CLIP_CONFIG_REVISION)
            )
            state = load_file(weights)
            state = {key: value for key, value in state.items() if not key.endswith(".position_ids")}
            self._clip_model.load_state_dict(state, strict=True)
            self._clip_model.to(self.config.device)
            self._clip_model.eval()
        except Exception as exc:  # noqa: BLE001
            self._clip_error = f"Failed to load modality model '{self.config.model_id}': {exc}"
            self._clip_model = None
            self._clip_processor = None
            return False
        return True

    def _clip_validate(self, image_path: str | Path, t0: float) -> ModalityResult:
        import time
        import torch.nn.functional as F

        model_id = self.config.model_id
        try:
            if Path(image_path).suffix.lower() in {".dcm", ".dicom"}:
                from chest_xpert.preprocessing import load_image
                arr = load_image(image_path)
                image = Image.fromarray(np.uint8(np.clip((arr + 1024) / 2048, 0, 1) * 255)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=f"Cannot open image for CLIP modality check: {exc}",
                model_id=model_id,
                technical_error=str(exc),
                method="clip_zero_shot",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if not self._ensure_clip_loaded():
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=self._clip_error or "CLIP model unavailable.",
                model_id=model_id,
                technical_error=self._clip_error,
                method="clip_zero_shot",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            inputs = self._clip_processor(
                text=self.config.candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = F.softmax(logits_per_image, dim=-1)[0]

            probs_dict = {
                label: float(p)
                for label, p in zip(self.config.candidate_labels, probs)
            }
            cxr_prob = probs_dict.get("a chest X-ray", 0.0)
            others = {
                k: v for k, v in probs_dict.items() if k != "a chest X-ray"
            }
            best_other_label = max(others, key=others.get) if others else None
            best_other_prob = others[best_other_label] if best_other_label else 0.0

            cfg = self.config
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if (
                cxr_prob >= cfg.accepted_min_prob
                and cxr_prob > best_other_prob + cfg.accepted_margin_over_best_other
            ):
                decision = ModalityDecision.ACCEPTED
                reason = "Image appears to be a chest X-ray (CLIP zero-shot)."
            elif (
                cxr_prob <= cfg.rejected_max_prob
                and best_other_prob > cxr_prob + cfg.accepted_margin_over_best_other
            ):
                decision = ModalityDecision.REJECTED
                reason = "Image does not appear to be a chest X-ray."
            else:
                decision = ModalityDecision.UNCERTAIN
                reason = (
                    "Insufficient evidence to classify the image modality reliably."
                )

            return ModalityResult(
                decision=decision,
                chest_xray_prob=cxr_prob,
                best_other_label=best_other_label,
                best_other_prob=best_other_prob,
                all_probs=probs_dict,
                reason=reason,
                model_id=model_id,
                latency_ms=latency_ms,
                method="clip_zero_shot",
            )
        except Exception as exc:  # noqa: BLE001
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=f"CLIP inference failed: {exc}",
                model_id=model_id,
                technical_error=str(exc),
                method="clip_zero_shot",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_validate(self, image_path: str | Path, t0: float) -> ModalityResult:
        """Lightweight heuristic modality check used when both Qwen and CLIP fail.

        Chest X-rays are typically near-grayscale (low color saturation). This
        heuristic can reject obviously color photographs but cannot confirm an
        image is a chest X-ray. It is intentionally conservative: grayscale
        non-X-rays (documents, scans, drawings) fall through to 'uncertain'.

        NOT a validated clinical test.
        """
        import time

        model_id = self.config.model_id
        try:
            if Path(image_path).suffix.lower() in {".dcm", ".dicom"}:
                from chest_xpert.preprocessing import load_image
                arr = load_image(image_path)
                image = Image.fromarray(np.uint8(np.clip((arr + 1024) / 2048, 0, 1) * 255)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return ModalityResult(
                decision=ModalityDecision.TECHNICAL_FAILURE,
                reason=f"Cannot open image for modality check: {exc}",
                model_id=model_id,
                technical_error=str(exc),
                method="heuristic",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        arr = np.asarray(image).astype(np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return ModalityResult(
                decision=ModalityDecision.UNCERTAIN,
                reason="Image could not be assessed by heuristic (not RGB).",
                model_id=model_id,
                technical_error="non-rgb",
                method="heuristic",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        saturation = (
            np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        )
        mean_saturation = float(saturation.mean())

        # Heuristic threshold (documented heuristic, NOT validated).
        color_photo_saturation = 25.0
        if mean_saturation > color_photo_saturation:
            decision = ModalityDecision.REJECTED
            reason = (
                "Heuristic fallback: image has high color saturation and does not "
                "appear to be a grayscale chest X-ray."
            )
        else:
            decision = ModalityDecision.UNCERTAIN
            reason = (
                "Heuristic fallback: image is grayscale-like but neither Qwen nor "
                "CLIP is available, so chest X-ray modality cannot be confirmed."
            )

        return ModalityResult(
            decision=decision,
            chest_xray_prob=None,
            best_other_label=None,
            best_other_prob=None,
            all_probs=None,
            reason=reason,
            model_id=model_id + " [heuristic fallback]",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            method="heuristic",
            technical_error="Qwen and CLIP unavailable; heuristic used.",
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(self, image_path: str | Path) -> ModalityResult:
        """CLIP is the sole modality gate; unavailable/uncertain never passes."""
        import time
        return self._clip_validate(image_path, time.perf_counter())


# Share the loaded CLIP model between startup and requests.
modality_validator = ModalityValidator()
