"""Stage B — LLM: visual evidence review, prediction verification, and
report generation.

A single LLM (Qwen3-VL-32B when available) performs all three tasks:
  A. Image assessment — inspects the actual chest X-ray image and flags
     image quality / interpretation limitations.
  B. Prediction verification — per-pathology consistent / inconsistent /
     uncertain assessment compared against the classifier output.
  C. Report generation — produces a draft radiology report with sections
     for findings, impression, recommendations, and limitations.

The LLM is a secondary, model-generated assessment — NOT ground truth,
NOT a substitute for a radiologist, and never presented as proof of a
diagnosis.

If no LLM endpoint/token/model is available, verification and report
generation are marked unavailable — never fabricated.

Model priority (free on Hugging Face):
  1. Qwen/Qwen3-VL-32B-Instruct (vision-language, ~16 GB VRAM in 4-bit)
     — can actually see the X-ray image; best quality but needs GPU VRAM.
  2. Qwen/Qwen2.5-1.5B-Instruct (text-only, ~2 GB RAM) — reliable fallback
     on CPU; reasons from classifier scores + GradCAM description.
  3. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1 GB RAM) — smallest fallback.
  4. HF Inference API (remote) — if CHESTXPERT_HF_TOKEN is set.

Credentials come only from environment variables; nothing is hardcoded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests

from chest_xpert.config import LLM_CONFIG, LABELS_18, LLM_MODEL_CONFIG
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


def _build_prompt(
    image_available: bool,
    mappings: list[LabelMapping],
    model_doc: str,
) -> str:
    """Build a radiology-report prompt for the LLM.

    The prompt mirrors the HCT_LSR Qwen radiologist pattern:
    system policy + report sections (findings, impression, limitations).
    """
    lines = []
    lines.append("You are assisting a radiologist with chest X-ray interpretation.")
    lines.append("")
    lines.append("Your role is to assist a qualified radiologist. You do NOT replace "
                 "the radiologist and you must not present yourself as an "
                 "autonomous diagnostic authority.")
    lines.append("")
    lines.append("SCOPE:")
    lines.append("- You may discuss the provided chest X-ray.")
    lines.append("- You may discuss findings visible or relevant to the provided "
                 "chest X-ray.")
    lines.append("- You may discuss diseases, abnormalities, anatomy, imaging "
                 "findings, differential considerations, uncertainty, and "
                 "terminology when relevant to this chest X-ray.")
    lines.append("- You may explain the HCT-LSR model's predictions, "
                 "probabilities, confidence, and supported findings when those "
                 "data are provided.")
    lines.append("- You may explain why a prediction may have been produced using "
                 "the information available to you.")
    lines.append("- You may discuss the AI-generated report and the "
                 "radiologist-edited report for this case.")
    lines.append("")
    lines.append("OUT-OF-SCOPE:")
    lines.append("- Do not answer unrelated medical questions.")
    lines.append("- Do not answer questions about diseases unrelated to the "
                 "chest X-ray.")
    lines.append("- Do not provide general medical advice unrelated to this case.")
    lines.append("- If asked an unrelated question, politely state that you are "
                 "restricted to this chest X-ray case and related thoracic/"
                 "radiology discussion.")
    lines.append("")
    lines.append("REPORTING:")
    lines.append("- Generate a draft report that can be edited by the radiologist.")
    lines.append("- Clearly distinguish observations from interpretation.")
    lines.append("- Do not claim certainty when the image or model evidence is "
                 "uncertain.")
    lines.append("- Do not invent findings, clinical history, laboratory "
                 "results, or patient information.")
    lines.append("- The radiologist remains responsible for the final report.")
    lines.append("")
    lines.append("EDITED REPORT:")
    lines.append("- The radiologist may modify the AI-generated report.")
    lines.append("- After editing, treat the latest radiologist-edited report as "
                 "the authoritative report for discussion of the current case.")
    lines.append("- Continue answering questions after editing.")
    lines.append("- When answering after editing, explicitly distinguish the "
                 "edited report from the original AI draft when relevant.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("CLASSIFIER OUTPUT (pretrained DenseNet-121, torchxrayvision):")
    lines.append("=" * 70)
    for m in mappings:
        score_txt = f"{m.score:.4f}" if (m.supported and m.score is not None) \
            else "unsupported"
        lines.append(f"- {m.requested_label}: {score_txt} "
                     f"(supported={m.supported})")
    lines.append("")
    lines.append("Model documentation: " + model_doc)
    lines.append("")
    lines.append("TASKS:")
    lines.append("1. For each pathology, assess whether evidence is consistent "
                 "with the classifier prediction (consistent / inconsistent / "
                 "uncertain).")
    lines.append("2. Generate report findings ONLY where evidence supports a "
                 "finding (supported / not_supported / uncertain).")
    lines.append("")
    lines.append("For each supported pathology, output a verification item "
                 "with these fields:")
    lines.append("- pathology: name (string)")
    lines.append("- classifier_score: number or null")
    lines.append("- assessment: one of 'consistent', 'inconsistent', "
                 "'uncertain'")
    lines.append("- rationale: concise rationale based on available evidence "
                 "(string)")
    lines.append("- uncertainty: any uncertainty or limitations (string)")
    lines.append("- human_review_recommended: boolean")
    lines.append("")
    lines.append("Also output report_findings — only include pathologies where "
                 "the evidence collectively supports a finding. Each report "
                 "finding has:")
    lines.append("- pathology: name (string)")
    lines.append("- status: 'supported', 'not_supported', or 'uncertain'")
    lines.append("- evidence_basis: brief explanation of why (or why not) "
                 "supported")
    lines.append("- recommended_wording: brief phrase suitable for a radiology "
                 "report")
    lines.append("")
    lines.append("Also provide an image-level 'image_assessment' string "
                 "describing any interpretability or quality limitation "
                 "(or null if none).")
    lines.append("")
    lines.append("SAFEGUARDS:")
    lines.append("- Do not label a prediction definitively correct just "
                 "because its score is high.")
    lines.append("- Do not claim Grad-CAM or any heatmap proves a disease "
                 "is present.")
    lines.append("- Do not report a finding unless classifier + verification "
                 "+ GradCAM collectively support it.")
    lines.append("- If evidence is insufficient, return 'uncertain'.")
    lines.append("- Preserve the classifier output exactly; do not alter "
                 "scores or labels.")
    lines.append("- Do not fabricate observations not present in the image "
                 "or classifier output.")
    lines.append("- Do not invent patient information or examination details.")
    lines.append("")
    lines.append("Return ONLY a JSON object with keys 'image_assessment', "
                 "'items', and 'report_findings'. Example item:")
    lines.append('{"pathology": "Atelectasis", "assessment": "consistent", '
                 '"rationale": "...", "uncertainty": "...", '
                 '"human_review_recommended": true}.')
    return "\n".join(lines)


def _validate_and_build_items(
    parsed: dict,
    mappings: list[LabelMapping],
) -> tuple[list[VerificationItem], list[ReportFinding], list[str]]:
    """Validate LLM JSON into VerificationItem and ReportFinding lists.

    Report findings are derived deterministically from verification items:
    consistent + high score → supported; inconsistent → not_supported;
    uncertain → uncertain.
    """
    items: list[VerificationItem] = []
    findings: list[ReportFinding] = []
    warnings: list[str] = []
    supported_by_name = {
        m.requested_label: m for m in mappings if m.supported
    }
    # Normalize common LLM output variations to standard label names.
    # LLMs often use plural forms or slight variations (e.g. "Infiltrates"
    # instead of "Infiltration", "Pleural Effusion" instead of "Effusion").
    PATHOLOGY_SYNONYMS = {
        "infiltrates": "Infiltration",
        "infiltration": "Infiltration",
        "pleural effusion": "Effusion",
        "effusion": "Effusion",
        "atelectasis": "Atelectasis",
        "consolidation": "Consolidation",
        "pneumothorax": "Pneumothorax",
        "edema": "Edema",
        "pulmonary edema": "Edema",
        "emphysema": "Emphysema",
        "fibrosis": "Fibrosis",
        "pulmonary fibrosis": "Fibrosis",
        "pneumonia": "Pneumonia",
        "pleural thickening": "Pleural Thickening",
        "cardiomegaly": "Cardiomegaly",
        "nodule": "Nodule",
        "lung nodule": "Nodule",
        "mass": "Mass",
        "lung mass": "Mass",
        "hernia": "Hernia",
        "diaphragmatic hernia": "Hernia",
        "lung lesion": "Lung Lesion",
        "lesion": "Lung Lesion",
        "fracture": "Fracture",
        "rib fracture": "Fracture",
        "lung opacity": "Lung Opacity",
        "opacity": "Lung Opacity",
        "enlarged cardiomediastinum": "Enlarged Cardiomediastinum",
        "cardiomediastinum": "Enlarged Cardiomediastinum",
    }
    raw_items = parsed.get("items", [])
    if not isinstance(raw_items, list):
        warnings.append("'items' is not a list; verification treated as "
                        "unavailable.")
        return items, findings, warnings

    for entry in raw_items:
        if not isinstance(entry, dict):
            warnings.append("Skipping non-object item in verification output.")
            continue
        pathology = entry.get("pathology")
        if not pathology:
            warnings.append("Skipping item with missing pathology name.")
            continue
        # Normalize: case-insensitive lookup with synonym map.
        pathology_key = str(pathology).strip().lower()
        normalized = PATHOLOGY_SYNONYMS.get(pathology_key, pathology)
        if normalized not in supported_by_name:
            warnings.append(
                f"Skipping unsupported/unrecognized pathology: {pathology}")
            continue
        assessment = str(entry.get("assessment", "uncertain")).strip().lower()
        if assessment not in {a.value for a in Assessment}:
            assessment = Assessment.UNCERTAIN.value
            warnings.append(f"Invalid assessment for {pathology}; "
                            "defaulted to uncertain.")
        score = entry.get("classifier_score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
        mapping = supported_by_name[normalized]
        items.append(
            VerificationItem(
                pathology=normalized,
                classifier_score=(
                    score if score is not None else mapping.score
                ),
                assessment=assessment,
                rationale=str(entry.get("rationale",
                                        "No rationale provided.")),
                uncertainty=str(entry.get("uncertainty",
                                          "Not specified.")),
                human_review_recommended=bool(
                    entry.get("human_review_recommended", True)),
            )
        )

    # Derive report findings deterministically from verification items.
    for it in items:
        if it.assessment == Assessment.CONSISTENT.value and (
            it.classifier_score is not None
            and it.classifier_score >= 0.5
        ):
            status = "supported"
        elif it.assessment == Assessment.INCONSISTENT.value:
            status = "not_supported"
        else:
            status = "uncertain"
        findings.append(
            ReportFinding(
                pathology=it.pathology,
                status=status,
                evidence_basis=it.rationale,
                recommended_wording=(
                    f"{it.pathology}: {it.assessment} — "
                    f"{it.rationale[:80]}"
                ),
            )
        )

    return items, findings, warnings


class LLMClient:
    """Minimal remote LLM client for chat-style completion.

    Credentials are read from LLM_CONFIG (environment variables only). Nothing
    is hardcoded. All network calls carry a timeout. On any failure the client
    reports itself as unavailable rather than fabricating output.
    """

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
        """Call the LLM and return (parsed_json, error). Never raises."""
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
            content = data.get("choices", [{}])[0].get("message", {}).get(
                "content", "")
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
# Generic LLM support for Stage B (free Hugging Face options)
# ---------------------------------------------------------------------------


def _is_vlm_model(model_id: str) -> bool:
    """Heuristic: VLM models typically have 'VL' in the name."""
    return "VL" in model_id or "vl" in model_id


class LLMModelClient:
    """Client for LLM models on Hugging Face.

    Supports three backends, all free:
      * ``hf_api``   — Hugging Face Inference API (rate-limited free tier).
                       Needs CHESTXPERT_HF_TOKEN and internet access.
      * ``transformers_local`` — loads model locally via transformers.
                       Qwen3-VL-32B needs ~16 GB VRAM in 4-bit (fits on
                       32 GB V100).
                       Small text-only models (~0.5B–3B) run on CPU.
      * ``vllm_server`` — connect to a self-hosted vLLM server at the
                       configured URL (run vLLM separately).

    Credentials come only from environment variables; nothing is hardcoded.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
        backend: str = "transformers_local",
        force_cpu: bool = False,
        vllm_url: str = "",
        load_in_4bit: bool = True,
    ):
        self.model_id = model_id
        self.backend = backend
        self.force_cpu = force_cpu
        self.vllm_url = vllm_url
        self.load_in_4bit = load_in_4bit
        self._is_vlm = _is_vlm_model(model_id)
        self._token = None
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None

        if backend == "hf_api":
            import os

            self._token = os.environ.get("CHESTXPERT_HF_TOKEN")
        elif backend == "transformers_local":
            self._ensure_local_loaded()
        elif backend == "vllm_server":
            self._token = os.environ.get("CHESTXPERT_HF_TOKEN")
            # vLLM server is assumed to be running externally; nothing to
            # load locally.

    def is_configured(self) -> bool:
        if self.backend == "hf_api":
            return bool(self._token)
        if self.backend == "transformers_local":
            return self._model is not None and self._load_error is None
        if self.backend == "vllm_server":
            return bool(self.vllm_url)
        return False

    def _ensure_local_loaded(self) -> None:
        """Load model locally via transformers.

        Fallback chain:
          1. Configured model (Qwen3-VL-32B-Instruct per the notebook) —
             vision-language, can see the X-ray image.
          2. Qwen2-VL-2B-Instruct — small vision-language fallback that can
             still see the image, fits comfortably in available VRAM.
          3. Template fallback (no LLM) if both fail.
        """
        fallback_model = "Qwen/Qwen2-VL-2B-Instruct"

        try:
            self._load_single_model(self.model_id)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Primary model '{self.model_id}' failed: {exc}"
            # Try small VLM fallback (can see images, unlike text-only models).
            try:
                fallback_client = LLMModelClient(
                    model_id=fallback_model,
                    backend="transformers_local",
                    force_cpu=False,
                    load_in_4bit=False,
                )
                self._model = fallback_client._model
                self._processor = fallback_client._processor
                self.model_id = fallback_model
                self._is_vlm = _is_vlm_model(fallback_model)
                self.force_cpu = fallback_client.force_cpu
                self._load_error = None
            except Exception as exc2:  # noqa: BLE001
                self._load_error = (
                    f"Primary model '{self.model_id}' failed: {exc}; "
                    f"VLM fallback '{fallback_model}' also failed: {exc2}"
                )
                self._model = None
                self._processor = None

    def _load_single_model(self, model_id: str) -> None:
        """Load a single model via transformers.

        For Qwen VL models: loads on GPU (FP16 or 4-bit if bitsandbytes works).
        For small text-only models: runs on CPU.
        """
        from transformers import AutoProcessor
        import torch

        # Select the correct VLM model class based on model id.
        is_qwen3_vl = "Qwen3-VL" in model_id or "qwen3-vl" in model_id.lower()
        is_qwen2_vl = "Qwen2-VL" in model_id or "qwen2-vl" in model_id.lower()

        if is_qwen3_vl:
            from transformers import Qwen3VLForConditionalGeneration
            vlm_model_class = Qwen3VLForConditionalGeneration
        elif is_qwen2_vl:
            from transformers import Qwen2VLForConditionalGeneration
            vlm_model_class = Qwen2VLForConditionalGeneration
        else:
            # Generic VLM fallback — try Qwen3 class.
            from transformers import Qwen3VLForConditionalGeneration
            vlm_model_class = Qwen3VLForConditionalGeneration

        from transformers import BitsAndBytesConfig

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

        load_kwargs: dict = {}
        if (self.load_in_4bit
                and torch.cuda.is_available()
                and not self.force_cpu):
            try:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            except Exception:
                # bitsandbytes broken — fall back to FP16 on GPU.
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        self._model = vlm_model_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            **load_kwargs,
        )
        self._model.eval()

    def verify(
        self,
        mappings: list[LabelMapping],
        model_doc: str,
        image_path: Optional[str] = None,
    ) -> VerificationResult:
        """Run prediction verification + report generation in ONE LLM call.

        Returns structured VerificationResult with both verification items
        and report_findings.
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

        prompt = _build_prompt(
            image_available=image_path is not None,
            mappings=mappings,
            model_doc=model_doc,
        )

        try:
            if self.backend == "hf_api":
                parsed, err = self._call_hf_api(prompt)
            elif self.backend == "vllm_server":
                parsed, err = self._call_vllm_server(prompt)
            else:
                parsed, err = self._call_local(prompt, image_path)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if err or parsed is None:
                return VerificationResult(
                    items=[],
                    report_findings=[],
                    model_id=self.model_id,
                    available=False,
                    error=f"LLM verification failed: {err}",
                    latency_ms=latency_ms,
                )

            items, findings, warnings = _validate_and_build_items(
                parsed, mappings)
            image_assessment = parsed.get("image_assessment")
            return VerificationResult(
                items=items,
                report_findings=findings,
                model_id=self.model_id,
                available=True,
                error=("; ".join(warnings) if warnings else None),
                latency_ms=latency_ms,
                image_assessment=(
                    str(image_assessment)
                    if image_assessment is not None else None
                ),
                image_assessment_note=(
                    "LLM assessment (secondary, not ground truth)."
                    if image_assessment else None
                ),
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

    def _call_hf_api(self, prompt: str
                     ) -> tuple[Optional[dict], Optional[str]]:
        import os

        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.2, "max_new_tokens": 1024},
        }
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                content = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                content = data.get("generated_text", "")
            else:
                content = str(data)
            if not content:
                return None, "LLM API returned empty content."
            content = self._extract_json(content)
            import json as _json
            return _json.loads(content), None
        except requests.RequestException as exc:  # noqa: BLE001
            return None, f"LLM API request failed: {exc}"
        except (ValueError, KeyError) as exc:  # noqa: BLE001
            return None, f"LLM API returned malformed JSON: {exc}"

    def _call_vllm_server(
        self, prompt: str
    ) -> tuple[Optional[dict], Optional[str]]:
        """Call a self-hosted vLLM server (OpenAI-compatible API)."""
        import os

        url = self.vllm_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system",
                 "content": ("You are a careful medical imaging assistant. "
                             "Output valid JSON only.")},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get(
                "content", "")
            if not content:
                return None, "vLLM server returned empty content."
            content = self._extract_json(content)
            import json as _json
            return _json.loads(content), None
        except requests.RequestException as exc:  # noqa: BLE001
            return None, f"vLLM server request failed: {exc}"
        except (ValueError, KeyError) as exc:  # noqa: BLE001
            return None, f"vLLM server returned malformed JSON: {exc}"

    def _call_local(
        self, prompt: str, image_path: Optional[str]
    ) -> tuple[Optional[dict], Optional[str]]:
        try:
            import torch
            from PIL import Image
            import json as _json

            # Build messages following Qwen chat format
            messages = [
                {
                    "role": "system",
                    "content": ("You are a careful medical imaging "
                              "assistant. Output valid JSON only."),
                },
            ]

            # Build user content: image first (if VL), then text
            img = None
            if self._is_vlm and image_path:
                try:
                    img = Image.open(image_path).convert("RGB")
                    user_content = [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ]
                except Exception:
                    img = None
                    user_content = prompt
            else:
                # Text-only model: content is a plain string
                user_content = prompt

            messages.append({"role": "user", "content": user_content})

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if self._is_vlm and img is not None:
                inputs = self._processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=[text],
                    return_tensors="pt",
                )

            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device)
                          for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=1024,
                    temperature=0.2, do_sample=False)
                generated = self._processor.decode(
                    out[0], skip_special_tokens=True)

            content = self._extract_json(generated)
            return _json.loads(content), None
        except Exception as exc:  # noqa: BLE001
            return None, f"Local inference failed: {exc}"
        finally:
            # Clean up GPU memory after inference (matching HCT_LSR pattern)
            self._cleanup_gpu()

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull a JSON object out of text that may contain extra content."""
        import re

        # Try markdown code fence first
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Find the first complete JSON object by matching braces
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            return text[start:end + 1]
        return text

    @staticmethod
    def _cleanup_gpu() -> None:
        """Free GPU memory after inference (matching HCT_LSR pattern)."""
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class VerificationEngine:
    """Stage B engine. One LLM does verification + report generation."""

    def __init__(self, client=None):
        self.client = client or LLMModelClient()

    def verify(
        self,
        mappings: list[LabelMapping],
        model_doc: str,
        image_path: Optional[str] = None,
    ) -> VerificationResult:
        """Run prediction verification + report generation in one LLM call."""
        return self.client.verify(mappings, model_doc, image_path)
