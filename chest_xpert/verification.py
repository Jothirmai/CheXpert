"""Vision report generation from mandatory CNN scores, original X-ray and Grad-CAM.

Report inference requires a local CUDA vision-language model. Missing evidence or
image-processing failures stop report generation rather than producing a text-only
fallback. Template scanning remains a separate one-image operation. Reports are
model-generated drafts requiring radiologist review.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests

from chest_xpert.config import LLM_CONFIG, LABELS_18, LLM_MODEL_CONFIG, LORA_CONFIG
from chest_xpert.label_mapping import LabelMapping


class Assessment(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNCERTAIN = "uncertain"


@dataclass
class VerificationItem:
    pathology: str
    classifier_score: Optional[float]
    assessment: str  # consistent | inconsistent | uncertain
    rationale: str
    uncertainty: str
    human_review_recommended: bool


@dataclass
class ReportFinding:
    pathology: str
    status: str  # supported | not_supported | uncertain
    evidence_basis: str
    recommended_wording: str


@dataclass
class VerificationResult:
    items: list[VerificationItem] = field(default_factory=list)
    report_findings: list[ReportFinding] = field(default_factory=list)
    model_id: str = ""
    available: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0
    image_assessment: Optional[str] = None
    image_assessment_note: Optional[str] = None
    # V7 full report text
    v7_report: Optional[str] = None

    def item_for(self, pathology: str) -> Optional[VerificationItem]:
        for it in self.items:
            if it.pathology == pathology:
                return it
        return None

    def disagreements(self) -> list[VerificationItem]:
        return [it for it in self.items if it.assessment == Assessment.INCONSISTENT]

    def uncertain_items(self) -> list[VerificationItem]:
        return [it for it in self.items if it.assessment == Assessment.UNCERTAIN]


def _empty_result(reason: str, model_id: str = "", available: bool = False) -> VerificationResult:
    return VerificationResult(
        items=[],
        report_findings=[],
        model_id=model_id,
        available=available,
        error=reason,
    )


# ---------------------------------------------------------------------------
# V7 Report Format Prompt
# ---------------------------------------------------------------------------

def _build_v7_report_prompt(
    mappings: list[LabelMapping],
    model_doc: str,
    patient_info: dict,
    image_available: bool,
    contact_sheet_available: bool,
    few_shot_examples: Optional[list[dict]] = None,
) -> str:
    """Build the V7 Qwen3-VL All-Disease Radiologist Evidence Fusion report prompt.

    The VLM receives the original X-ray, Grad-CAM contact sheet, and classifier
    scores, and produces a structured radiology report.
    """
    headings = patient_info.get("report_sections", ["Clinical indication", "Comparison", "Technique", "Findings", "Impression"])
    evidence = {m.requested_label: m.score for m in mappings if m.supported}
    original_count = int(patient_info.get('original_image_count', 1))
    image_description = ("The second image is a Grad-CAM contact sheet for selected CNN findings. " if original_count == 1 else f"The first {original_count} images are original radiographs from the same study; the final image is a Grad-CAM contact sheet, not another projection. ")
    return (
        "You assist a radiologist reviewing a chest X-ray. Inspect the ORIGINAL image first. "
        + image_description + "Heatmaps "
        "are model attention, not localization proof. CNN scores are uncalibrated supporting "
        "signals, not disease probabilities, clinical severity or diagnoses. Never infer a "
        "finding just because a score is high or exclude a finding because it is low. "
        "All three evidence types are mandatory: original radiograph(s), labeled Grad-CAM contact sheet, and exact CNN scores. "
        "The CNN was trained with class weighting, so even several large scores can occur without matching image findings. "
        "Use scores to decide what to inspect and the contact sheet to see where the CNN attended. "
        "Confirm each proposed finding in the original image; if support is uncertain, retain that uncertainty. "
        "Do not turn a bright heatmap into a clinical finding. Consider projection and lung volume before judging heart size. "
        "Ground observations in the original image, acknowledge limitations and uncertainty. "
        "In Findings address the cardiomediastinal silhouette, lung parenchyma, pleural spaces, "
        "diaphragms and visible bones/devices as assessable; do not invent normal findings for "
        "poorly seen structures. In Impression prioritize significant findings without "
        "unnecessary repetition. Never include publisher names, author names, journal titles, "
        "article IDs, DOI, URLs, license/copyright text, dataset titles or citations. "
        "Do not invent patient information, history, comparisons or measurements. Use NOT PROVIDED "
        "for unavailable case information. In Clinical indication and Comparison sections use only "
        "supplied history/comparison metadata, never substitute imaging findings. Do not invent "
        "positioning or projection. Describe visible abnormalities specifically; avoid generic "
        "normal reports. Distinguish low lung volumes and technical limitations from disease. "
        "Do not write model names, scores or heatmap descriptions into clinical sections. "
        "Write Findings and Impression in the style of a conventional radiology report: concise clinical prose describing anatomy and radiographic observations only. "
        "Never mention Grad-CAM, heatmaps, saliency, attention maps, CNN, VLM, model outputs, classifier scores or how the report was generated in any report section. "
        "Do not add a methodological explanation or technical disclaimer to the report. "
        "Return ONLY valid JSON, no markdown, with this schema: "
        '{"sections":[{"heading":"exact requested heading","text":"report section content"}]}. '
        "Use exactly these headings in order: " + json.dumps(headings) +
        " Patient information: " + json.dumps(patient_info) +
        " Supporting CNN scores: " + json.dumps(evidence) +
        " Model: " + model_doc
    )



def build_report_messages(prompt, images):
    """Shared, ordered multimodal chat format for training and inference."""
    if len(images) < 2:
        raise ValueError("Report messages require original image(s) and Grad-CAM contact sheet")
    if len(images) > 2:
        content=[]
        for i,image in enumerate(images[:-1]):
            content.extend([{"type":"text","text":f"Original study radiograph {i+1}/{len(images)-1}."},{"type":"image","image":image}])
        content.extend([{"type":"text","text":"Final image: labeled Grad-CAM contact sheet, not a clinical projection."},{"type":"image","image":images[-1]},{"type":"text","text":prompt}])
        return [{"role":"system","content":"You assist a radiologist. Inspect all original study views. Follow the requested JSON schema; output clinical report JSON only."},{"role":"user","content":content}]
    return [
        {"role":"system", "content":"You assist a radiologist. Follow the requested JSON schema exactly. Use both images as instructed; output JSON only."},
        {"role":"user", "content":[
            {"type":"text", "text":"Image 1: original chest radiograph."},
            {"type":"image", "image":images[0]},
            {"type":"text", "text":"Image 2: labeled CNN Grad-CAM contact sheet; attention is not proof of disease."},
            {"type":"image", "image":images[1]},
            {"type":"text", "text":prompt},
        ]},
    ]


# ---------------------------------------------------------------------------
# LLM client (remote API)
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal remote LLM client for chat-style completion."""

    def __init__(self, config=None):
        self.config = config or LLM_CONFIG
        self._session = requests.Session()
        if self.config.api_token:
            self._session.headers.update(
                {"Authorization": f"Bearer {self.config.api_token}"}
            )

    def is_configured(self) -> bool:
        return bool(self.config.enabled and self.config.api_url)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> tuple[Optional[dict], Optional[str]]:
        if not self.is_configured():
            return None, "LLM not configured: CHESTXPERT_LLM_API_URL not set."
        payload = {
            "model": self.config.model_id or "default",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = self._session.post(
                self.config.api_url,
                json=payload,
                timeout=self.config.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return None, "LLM returned empty content."
            parsed = json.loads(content)
            return parsed, None
        except requests.RequestException as exc:  # noqa: BLE001
            return None, f"LLM request failed: {exc}"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:  # noqa: BLE001
            return None, f"LLM returned malformed JSON: {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, f"LLM error: {exc}"


# ---------------------------------------------------------------------------
# Local VLM client (Qwen3-VL on GPU)
# ---------------------------------------------------------------------------


def _is_vlm_model(model_id: str) -> bool:
    return "VL" in model_id or "vl" in model_id


class LLMModelClient:
    """Client for LLM models on Hugging Face.

    Supports three backends:
      * hf_api — Hugging Face Inference API (needs CHESTXPERT_HF_TOKEN).
      * transformers_local — loads model locally via transformers.
        Qwen3-VL-32B loads on GPU in 4-bit (~16 GB VRAM).
        Small text models run on CPU.
      * vllm_server — connect to a self-hosted vLLM server.
    """

    def __init__(
        self,
        model_id: str | None = None,
        backend: str | None = None,
        force_cpu: bool | None = None,
        vllm_url: str | None = None,
        load_in_4bit: bool | None = None,
        lora_adapter_path: str | None = None,
    ):
        from chest_xpert.config import LLM_MODEL_CONFIG, LORA_CONFIG

        self.model_id = model_id or LLM_MODEL_CONFIG.model_id
        self.backend = backend or LLM_MODEL_CONFIG.backend
        self.force_cpu = LLM_MODEL_CONFIG.force_cpu if force_cpu is None else force_cpu
        self.vllm_url = vllm_url or LLM_MODEL_CONFIG.vllm_url
        self.load_in_4bit = LLM_MODEL_CONFIG.load_in_4bit if load_in_4bit is None else load_in_4bit
        self.lora_adapter_path = lora_adapter_path or (LORA_CONFIG.adapter_path if LORA_CONFIG.enabled else None)
        self._is_vlm = _is_vlm_model(self.model_id)
        self._token = None
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None
        self._lora_loaded: bool = False

        # Fallback chain: VLM -> text-only
        self._fallback_models = []

        if self.backend == "hf_api":
            import os
            self._token = os.environ.get("CHESTXPERT_HF_TOKEN")
        elif self.backend == "transformers_local":
            self._ensure_local_loaded()
        elif self.backend == "vllm_server":
            import os
            self._token = os.environ.get("CHESTXPERT_HF_TOKEN")

    def is_configured(self) -> bool:
        if self.backend == "hf_api":
            return bool(self._token)
        if self.backend == "transformers_local":
            return self._model is not None and self._load_error is None
        if self.backend == "vllm_server":
            return bool(self.vllm_url)
        return False

    def _ensure_local_loaded(self) -> None:
        try:
            self._load_single_model(self.model_id)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Primary model '{self.model_id}' failed: {exc}"
            for fallback_model in self._fallback_models:
                try:
                    fallback_client = LLMModelClient(
                        model_id=fallback_model,
                        backend="transformers_local",
                        force_cpu=not _is_vlm_model(fallback_model),
                        load_in_4bit=_is_vlm_model(fallback_model),
                    )
                    self._model = fallback_client._model
                    self._processor = fallback_client._processor
                    self.model_id = fallback_model
                    self._is_vlm = fallback_client._is_vlm
                    self.force_cpu = fallback_client.force_cpu
                    self.load_in_4bit = fallback_client.load_in_4bit
                    self._load_error = None
                    return
                except Exception as exc2:  # noqa: BLE001
                    continue
            self._load_error = (
                f"Primary model '{self.model_id}' failed: {exc}; "
                f"all fallbacks also failed."
            )
            self._model = None
            self._processor = None

    def _load_single_model(self, model_id: str) -> None:
        from transformers import AutoProcessor
        import torch

        is_qwen3_vl = "Qwen3-VL" in model_id or "qwen3-vl" in model_id.lower()
        is_qwen25_vl = "qwen2.5-vl" in model_id.lower()
        is_qwen2_vl = "qwen2-vl" in model_id.lower()

        if is_qwen3_vl:
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
            tokenizer_class = AutoProcessor
            use_fast = True
        elif is_qwen25_vl:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
            tokenizer_class = AutoProcessor
            use_fast = True
        elif is_qwen2_vl:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
            tokenizer_class = AutoProcessor
            use_fast = False
        else:
            # Text-only model.
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_class = AutoModelForCausalLM
            tokenizer_class = AutoTokenizer
            use_fast = True

        self._processor = tokenizer_class.from_pretrained(
            model_id, trust_remote_code=True, use_fast=use_fast, min_pixels=256*28*28, max_pixels=768*28*28
        )

        if not torch.cuda.is_available() or self.force_cpu:
            raise RuntimeError("CUDA is required for local vision inference.")
        if not self._is_vlm:
            raise RuntimeError("A vision-language model is required; text-only models cannot inspect X-rays.")
        load_kwargs: dict = {}
        if (self.load_in_4bit
                and torch.cuda.is_available()
                and not self.force_cpu):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = {"": __import__("os").environ.get("CHESTXPERT_DEVICE", "cuda:0")}
            except Exception:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = {"": __import__("os").environ.get("CHESTXPERT_DEVICE", "cuda:0")}
        elif self.force_cpu or not torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = None
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = {"": __import__("os").environ.get("CHESTXPERT_DEVICE", "cuda:0")}

        self._model = model_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            **load_kwargs,
        )
        self._model.eval()
        model_device = next(self._model.parameters()).device
        print(f"[Chest Xpert] Qwen LLM model '{model_id}' loaded on device: {model_device}")
        # Load LoRA adapter if configured.
        if self.lora_adapter_path:
            self._load_lora_adapter(self.lora_adapter_path)

    def _load_lora_adapter(self, adapter_path: str) -> None:
        """Load a LoRA (PEFT) adapter on top of the base model.

        The base model stays frozen; only the low-rank adapter weights are
        applied at inference. Requires the ``peft`` package.
        """
        try:
            from peft import PeftModel, PeftConfig
            adapter_config = PeftConfig.from_pretrained(adapter_path)
            expected = adapter_config.base_model_name_or_path
            actual = {self.model_id, getattr(self._model.config, "_name_or_path", "")}
            if expected and expected not in actual:
                raise ValueError(f"Adapter requires base model {expected}; configured model is {self.model_id}")
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, trust_remote_code=True
            )
            self._model.eval()
            self._lora_loaded = True
            print(f"[Chest Xpert] LoRA adapter loaded from: {adapter_path}")
        except ImportError:
            self._load_error = (
                f"LoRA adapter path '{adapter_path}' set but 'peft' is not "
                "installed. Install with: pip install peft"
            )
            print(f"[Chest Xpert] {self._load_error}")
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load LoRA adapter '{adapter_path}': {exc}"
            print(f"[Chest Xpert] {self._load_error}")

    # ------------------------------------------------------------------
    # V7 report generation
    # ------------------------------------------------------------------

    def generate_v7_report(
        self,
        mappings: list[LabelMapping],
        model_doc: str,
        patient_info: dict,
        image_path: Optional[str] = None,
        contact_sheet_path: Optional[str] = None,
        few_shot_examples: Optional[list[dict]] = None,
        self_correct: bool = True,
    ) -> VerificationResult:
        """Generate the V7 Qwen3-VL evidence-fusion radiology report.

        Input to VLM: original X-ray, Grad-CAM contact sheet, classifier scores,
        and patient/case information. Optionally runs a self-correction pass
        (2nd LLM review) to catch missed / hallucinated findings.
        """
        import time
        t0 = time.perf_counter()

        if not self.is_configured():
            return VerificationResult(
                items=[],
                report_findings=[],
                model_id=self.model_id,
                available=False,
                error=(
                    "LLM not configured. Set CHESTXPERT_HF_TOKEN for hf_api, "
                    "or ensure sufficient RAM/VRAM for transformers_local, "
                    "or set CHESTXPERT_VLLM_URL for vllm_server."
                ),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        original_paths = list(image_path) if isinstance(image_path, (list, tuple)) else ([image_path] if image_path else [])
        patient_info = {**patient_info, "original_image_count": len(original_paths)}
        prompt = _build_v7_report_prompt(
            mappings=mappings,
            model_doc=model_doc,
            patient_info=patient_info,
            image_available=image_path is not None or contact_sheet_path is not None,
            contact_sheet_available=contact_sheet_path is not None,
            few_shot_examples=few_shot_examples,
        )

        from pathlib import Path
        import math
        required_scores = [m for m in mappings if m.supported]
        if self.backend != "transformers_local":
            return VerificationResult(items=[], report_findings=[], model_id=self.model_id, available=False,
                error="Report generation requires the local multimodal backend with both images and CNN scores.", latency_ms=0)
        try:
            valid_scores = bool(required_scores) and len({m.requested_label for m in required_scores}) == len(required_scores) and all(m.score is not None and math.isfinite(float(m.score)) and 0 <= float(m.score) <= 1 for m in required_scores)
        except (TypeError, ValueError):
            valid_scores = False
        if not valid_scores:
            return VerificationResult(items=[], report_findings=[], model_id=self.model_id, available=False,
                error="Finite CNN scores between zero and one are required for report generation.", latency_ms=0)
        originals = list(image_path) if isinstance(image_path, (list, tuple)) else ([image_path] if image_path else [])
        if not originals or not contact_sheet_path or any(not p or not Path(p).is_file() for p in originals) or not Path(contact_sheet_path).is_file():
            return VerificationResult(items=[], report_findings=[], model_id=self.model_id, available=False,
                error="Both the original X-ray and the Grad-CAM contact sheet are required for report generation.", latency_ms=0)
        resolved=[Path(p).resolve() for p in originals]+[Path(contact_sheet_path).resolve()]
        if len(set(resolved)) != len(resolved):
            return VerificationResult(items=[], report_findings=[], model_id=self.model_id, available=False,
                error="Original X-ray and Grad-CAM contact sheet must be separate images.", latency_ms=0)

        try:
            if self.backend == "hf_api":
                text = self._call_hf_api_generate(prompt)
            elif self.backend == "vllm_server":
                text = self._call_vllm_server_generate(prompt)
            else:
                text = self._call_local_generate(
                    prompt, image_path, contact_sheet_path
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if text is None:
                return VerificationResult(
                    items=[],
                    report_findings=[],
                    model_id=self.model_id,
                    available=False,
                    error=getattr(self, "_generation_error", None) or "LLM report generation returned empty content.",
                    latency_ms=latency_ms,
                )

            # ---------- Self-correction pass (2nd LLM review) ----------
            if self_correct and self.backend == "transformers_local":
                try:
                    corrected = self._self_correct_report(
                        draft_report=text,
                        mappings=mappings,
                        model_doc=model_doc,
                        image_path=image_path,
                        contact_sheet_path=contact_sheet_path,
                    )
                    if corrected:
                        text = corrected
                except Exception:  # noqa: BLE001
                    pass  # keep original draft if self-correction fails

            return VerificationResult(
                items=[],
                report_findings=[],
                model_id=self.model_id,
                available=True,
                error=None,
                latency_ms=latency_ms,
                v7_report=text,
            )
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(
                items=[],
                report_findings=[],
                model_id=self.model_id,
                available=False,
                error=f"LLM error: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # ------------------------------------------------------------------
    # Backend call helpers (V7 text generation)
    # ------------------------------------------------------------------

    def _call_hf_api_generate(self, prompt: str) -> Optional[str]:
        import os
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.2, "max_new_tokens": 2048},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", "")
            return str(data)
        except requests.RequestException as exc:  # noqa: BLE001
            return None

    def _call_vllm_server_generate(self, prompt: str) -> Optional[str]:
        import os
        url = self.vllm_url.rstrip("/") + "/v1/completions"
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("text", "")
        except requests.RequestException as exc:  # noqa: BLE001
            return None

    def _call_local_generate(
        self,
        prompt: str,
        image_path: Optional[str],
        contact_sheet_path: Optional[str],
    ) -> Optional[str]:
        self._generation_error = None
        self._last_vision_inputs = None
        self._last_generation_output = None
        self._last_report_prompt = prompt
        try:
            import torch
            from pathlib import Path
            from PIL import Image

            # Report generation validates both paths first. Template scanning may use one image.
            images: list = []
            original_paths = list(image_path) if isinstance(image_path, (list, tuple)) else ([image_path] if image_path else [])
            supplied_paths = original_paths + ([contact_sheet_path] if contact_sheet_path else [])
            seen = set()
            for p in supplied_paths:
                if p and p not in seen:
                    if not Path(p).is_file():
                        raise ValueError(f"Vision input is missing: {p}")
                    seen.add(p)
                    with Image.open(p) as opened:
                        images.append(opened.convert("RGB"))

            # For VLM models with at least one image, use chat template (multimodal).
            # For text-only models, use raw prompt to avoid role-marker echo.
            if self._is_vlm and images:
                try:
                    content = [
                        {"type": "image", "image": img} for img in images
                    ] + [{"type": "text", "text": prompt}]
                    messages = [
                        {
                            "role": "system",
                            "content": ("You assist a radiologist. Follow the requested JSON schema exactly. "
                                        "Use both images as instructed; output JSON only."),
                        },
                        {"role": "user", "content": content},
                    ]
                    if contact_sheet_path and len(images) >= 2:
                        messages = build_report_messages(prompt, images)
                    text = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=[text], images=images, return_tensors="pt"
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to prepare both vision inputs: {exc}") from exc
            else:
                # Text-only path: raw prompt, no chat template.
                text = prompt
                inputs = self._processor(
                    text=text, return_tensors="pt"
                )

            if self._is_vlm and images:
                image_grid = inputs.get("image_grid_thw")
                if image_grid is None or int(image_grid.shape[0]) != len(images):
                    raise RuntimeError("Vision processor did not retain every supplied image")
                import hashlib
                self._last_vision_inputs = {
                    "image_count": len(images),
                    "image_roles": (["original_xray"] * len(original_paths) + ["gradcam_contact_sheet"]) if contact_sheet_path else ["template_scan"] * len(images),
                    "original_image_count": len(original_paths) if contact_sheet_path else 0,
                    "image_paths": [str(Path(p).resolve()) for p in supplied_paths],
                    "image_sha256": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in supplied_paths],
                    "image_grid_thw": image_grid.tolist(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            model_device = next(self._model.parameters()).device
            moved: dict = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    moved[key] = value.to(model_device)
                else:
                    moved[key] = value
            inputs = moved

            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=768, max_time=100,
                    do_sample=False)
            generated = self._processor.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Strip the prompt echo robustly.
            generated = generated.strip()
            if generated.startswith(text):
                generated = generated[len(text):].strip()
            else:
                # VLM chat-template prefixes the assistant response with a role
                # marker; find the last occurrence and take everything after it.
                for marker in ("\nassistant\n", "assistant\n", "\nassistant "):
                    pos = generated.rfind(marker)
                    if pos != -1:
                        generated = generated[pos + len(marker):].strip()
                        break
                else:
                    # Last resort: find actual report content (only in response).
                    for marker in ("Patient ID:", "Study: Chest X-ray"):
                        pos = generated.rfind(marker)
                        if pos != -1:
                            generated = generated[pos:].strip()
                            break

            self._last_generation_output = generated
            return generated
        except Exception as exc:  # noqa: BLE001
            self._generation_error = f"Vision generation failed: {exc}"
            return None
        finally:
            self._cleanup_gpu()

    @staticmethod
    def _cleanup_gpu() -> None:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------------
    # Self-correction pass (2nd LLM review of the draft report)
    # ------------------------------------------------------------------

    def _self_correct_report(self, draft_report, mappings, model_doc, image_path, contact_sheet_path):
        """Optional second pass retaining exactly the same three evidence inputs."""
        scores = {m.requested_label:m.score for m in mappings if m.supported}
        prompt = (
            "Review this draft using BOTH supplied images: first the original chest radiograph, "
            "then the labeled Grad-CAM contact sheet. CNN scores below are uncalibrated supporting "
            "signals, not diagnoses. Preserve uncertainty and technical limitations. Remove claims "
            "unsupported by the original image and retain directly visible findings. "
            "Do not put model, score or heatmap descriptions in clinical sections. "
            "Preserve the requested section headings. Return ONLY JSON with schema "
            '{"sections":[{"heading":"heading","text":"clinical content"}]}. '
            "Supporting CNN scores: " + json.dumps(scores) +
            " Model context: " + model_doc + " Draft: " + draft_report
        )
        return self._call_local_generate(prompt, image_path, contact_sheet_path)


# ---------------------------------------------------------------------------
# Verification engine (Stage B)
# ---------------------------------------------------------------------------

class VerificationEngine:
    """Stage B engine. One LLM does verification + V7 report generation."""

    def __init__(self, client=None):
        self.client = client or LLMModelClient()

    def verify(
        self,
        mappings: list[LabelMapping],
        model_doc: str,
        image_path: Optional[str] = None,
        contact_sheet_path: Optional[str] = None,
    ) -> VerificationResult:
        """Run prediction verification + V7 report generation in one LLM call."""
        return self.client.verify(
            mappings, model_doc, image_path, contact_sheet_path
        )

    def generate_v7_report(
        self,
        mappings: list[LabelMapping],
        model_doc: str,
        patient_info: dict,
        image_path: Optional[str] = None,
        contact_sheet_path: Optional[str] = None,
        few_shot_examples: Optional[list[dict]] = None,
        self_correct: bool = True,
    ) -> VerificationResult:
        """Generate the V7 Qwen3-VL evidence-fusion report."""
        return self.client.generate_v7_report(
            mappings, model_doc, patient_info, image_path, contact_sheet_path,
            few_shot_examples=few_shot_examples, self_correct=self_correct,
        )


def _validate_and_build_items(parsed, mappings):
    """Parse legacy verification responses without converting scores to diagnoses."""
    supported = {m.requested_label: m for m in mappings if m.supported}
    items, warnings = [], []
    entries = parsed.get("items", [])
    if not isinstance(entries, list):
        return [], [], ["items must be a list"]
    for entry in entries:
        if not isinstance(entry, dict):
            warnings.append("Skipping non-object item")
            continue
        name = entry.get("pathology")
        if name not in supported:
            warnings.append(f"Skipping unsupported pathology: {name}")
            continue
        assessment = entry.get("assessment", "uncertain")
        if assessment not in {a.value for a in Assessment}:
            assessment = "uncertain"
            warnings.append(f"Invalid assessment for {name}; defaulted to uncertain")
        items.append(VerificationItem(name, supported[name].score, assessment,
            str(entry.get("rationale", "")), str(entry.get("uncertainty", "")), True))
    return items, [], warnings
