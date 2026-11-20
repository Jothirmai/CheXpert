"""Stage 2: GraphRAG-guided retrieval + LLM V7 radiology report generation.

Pipeline:
  1. Build a clinical knowledge graph from training reports.
  2. Retrieve graph nodes relevant to Stage 1 disease predictions.
  3. LLM restructures retrieved evidence into a V7
     "Qwen3-VL All-Disease Radiologist Evidence Fusion" report.

No LLM fallback — raises RuntimeError if no LLM is configured.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# V7 report data model
# ---------------------------------------------------------------------------

@dataclass
class PatientInfo:
    patient_id: str = "NOT PROVIDED"
    study: str = "Chest X-ray"
    view: str = "Posteroanterior (PA)"
    patient_age: str = "NOT PROVIDED"
    patient_sex: str = "NOT PROVIDED"
    clinical_indication: str = "NOT PROVIDED"
    comparison_study: str = "NOT PROVIDED"


@dataclass
class AIEvidence:
    disease: str
    probability: float
    gradcam_location: str = ""
    supported: bool = False
    notes: str = ""


@dataclass
class V7Report:
    patient: PatientInfo
    exam_technique: str
    image_quality: str
    findings: dict[str, str]           # subsection letter -> text
    ai_evidence: str                   # section 5 text
    impression: str                    # section 6 text
    full_text: str                     # assembled V7 report
    diseases: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clinical knowledge graph (unchanged from QA version, reused for retrieval)
# ---------------------------------------------------------------------------

class ClinicalKnowledgeGraph:
    """Builds and queries a clinical knowledge graph from radiology reports."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.finding_patterns = [
            r"atelectasis", r"cardiomegaly", r"consolidation", r"edema",
            r"effusion", r"emphysema", r"fibrosis", r"hernia", r"infiltration",
            r"mass", r"nodule", r"pleural thickening", r"pneumonia",
            r"pneumothorax",
        ]
        self.negation_patterns = [
            r"no\s+", r"without\s+", r"absence\s+of\s+", r"denies\s+",
            r"not\s+", r"clear\s+of\s+", r"unremarkable",
        ]
        self.uncertainty_patterns = [
            r"possible\s+", r"probable\s+", r"suspicious\s+for\s+",
            r"questionable\s+", r"rule\s+out\s+", r"cannot\s+exclude\s+",
        ]

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        text_lower = text.lower()
        findings = []
        for pat in self.finding_patterns:
            if re.search(pat, text_lower):
                findings.append(pat.strip(r"\\").replace(r"\s+", " ").title())
        is_negated = any(re.search(p, text_lower) for p in self.negation_patterns)
        is_uncertain = any(re.search(p, text_lower) for p in self.uncertainty_patterns)
        return {
            "findings": findings,
            "negated": is_negated,
            "uncertain": is_uncertain,
        }

    def add_report(self, report_id: str, report_text: str):
        entities = self._extract_entities(report_text)
        self.graph.add_node(report_id, type="report", text=report_text[:500])

        for finding in entities["findings"]:
            node_id = f"finding:{finding}"
            self.graph.add_node(node_id, type="finding", name=finding)
            self.graph.add_edge(report_id, node_id, relation="mentions")
            if entities["negated"]:
                self.graph.add_edge(node_id, report_id, relation="negated_in")
            if entities["uncertain"]:
                self.graph.add_edge(node_id, report_id, relation="uncertain_in")

        findings = entities["findings"]
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                n1 = f"finding:{findings[i]}"
                n2 = f"finding:{findings[j]}"
                self.graph.add_edge(n1, n2, relation="co_occurs")

    def retrieve(self, diseases: list[str], max_nodes: int = 10) -> list[dict]:
        results = []
        for disease in diseases:
            node_id = f"finding:{disease}"
            if node_id in self.graph:
                neighbors = list(self.graph.neighbors(node_id))
                results.append({
                    "disease": disease,
                    "node": node_id,
                    "neighbors": neighbors[:max_nodes],
                    "relations": [
                        self.graph[node_id][n].get("relation", "")
                        for n in neighbors[:max_nodes]
                    ],
                })
        return results


class GraphRAGRetriever:
    """Retrieves clinical evidence from the knowledge graph for Stage 2."""

    def __init__(self, kg: ClinicalKnowledgeGraph):
        self.kg = kg

    def retrieve_evidence(self, diseases: list[str]) -> dict[str, list[str]]:
        evidence = defaultdict(list)
        for d in diseases:
            results = self.kg.retrieve([d], max_nodes=5)
            for r in results:
                evidence[d].append(
                    f"{r['disease']}: {', '.join(r['neighbors'][:3])}"
                )
        return dict(evidence)


# ---------------------------------------------------------------------------
# V7 LLM client — generates V7-format reports (no Q/A fallback)
# ---------------------------------------------------------------------------

class V7LLMClient:
    """LLM client that generates V7 "All-Disease Radiologist Evidence Fusion"
    reports.

    Uses a local transformers model (text-only or VLM) or remote API.
    Loads on CUDA by default when available for low-latency inference.
    Supports optional LoRA (PEFT) adapters for fast fine-tuned inference
    without modifying base weights.
    Raises RuntimeError if no LLM is configured — no silent fallback.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        load_in_4bit: bool = True,
        lora_adapter_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.api_url = api_url
        self.api_token = api_token
        self.load_in_4bit = load_in_4bit
        self.lora_adapter_path = lora_adapter_path
        self._lora_loaded = False
        self._is_vlm = "VL" in model_id or "vl" in model_id.lower()

        # Resolve device: pick CUDA GPU with most free memory (Tesla V100 32GB).
        if device is None:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
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
                self.device = f"cuda:{best_idx}"
                print(f"[CliQ-RRG Stage 2] Selected GPU {best_idx} ({best_free / 1024**3:.1f} GB free)")
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[CliQ-RRG Stage 2] V7LLMClient device: {self.device} (VLM={self._is_vlm})")
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            print(f"[CliQ-RRG Stage 2] GPU: {torch.cuda.get_device_name(gpu_idx)}")

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._configured = bool(api_url) or self._try_local()

    def _try_local(self) -> bool:
        try:
            import torch
            from transformers import AutoTokenizer, BitsAndBytesConfig

            is_qwen2_vl = ("Qwen2-VL" in self.model_id or "qwen2-vl" in self.model_id.lower()
                           or "Qwen2.5-VL" in self.model_id or "qwen2.5-vl" in self.model_id.lower())
            is_qwen3_vl = "Qwen3-VL" in self.model_id or "qwen3-vl" in self.model_id.lower()

            if is_qwen2_vl or is_qwen3_vl:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                model_class = AutoModelForVision2Seq
                processor_class = AutoProcessor
                self._processor = processor_class.from_pretrained(
                    self.model_id, local_files_only=True, trust_remote_code=True
                )
                self._tokenizer = self._processor.tokenizer
            else:
                from transformers import AutoModelForCausalLM
                model_class = AutoModelForCausalLM
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, local_files_only=True
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            load_kwargs: dict = {}
            use_cuda = self.load_in_4bit and self.device.startswith("cuda") and torch.cuda.is_available()
            if use_cuda:
                try:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    # Force model onto the GPU with most free memory (V100 32GB).
                    load_kwargs["device_map"] = {"": self.device}
                except Exception:
                    load_kwargs["torch_dtype"] = "auto"
                    load_kwargs["device_map"] = {"": self.device}
            else:
                load_kwargs["torch_dtype"] = "auto"
                load_kwargs["device_map"] = {"": self.device}

            self._model = model_class.from_pretrained(
                self.model_id,
                local_files_only=True,
                trust_remote_code=True,
                **load_kwargs,
            )
            self._model.eval()

            model_device = next(self._model.parameters()).device
            print(f"[CliQ-RRG Stage 2] Base model '{self.model_id}' loaded on device: {model_device}")

            # Load LoRA adapter if configured.
            if self.lora_adapter_path:
                self._load_lora_adapter(self.lora_adapter_path)

            return True
        except Exception as e:
            print(f"Local LLM not available: {e}")
            return False

    def _load_lora_adapter(self, adapter_path: str) -> None:
        """Load a LoRA (PEFT) adapter on top of the base model.

        Base weights stay frozen; only low-rank adapter matrices are applied
        at inference. Output behaviour is preserved for inputs outside the
        fine-tuning distribution while task-specific responses improve.
        """
        try:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, trust_remote_code=True
            )
            self._model.eval()
            self._lora_loaded = True
            print(f"[CliQ-RRG Stage 2] LoRA adapter loaded from: {adapter_path}")
        except ImportError:
            print(f"[CliQ-RRG Stage 2] WARNING: LoRA adapter path '{adapter_path}' set "
                  f"but 'peft' is not installed. Install with: pip install peft")
        except Exception as exc:  # noqa: BLE001
            print(f"[CliQ-RRG Stage 2] WARNING: Failed to load LoRA adapter "
                  f"'{adapter_path}': {exc}")

    @property
    def configured(self) -> bool:
        return self._configured

    def generate_v7_report(
        self,
        patient: PatientInfo,
        diseases: list[str],
        disease_probs: dict[str, float],
        evidence: dict[str, list[str]],
        ai_evidence_items: list[AIEvidence],
        image_summary: str = "",
        image_path: Optional[str] = None,
        contact_sheet_path: Optional[str] = None,
    ) -> V7Report:
        """Generate a V7-format radiology report.

        Raises RuntimeError if no LLM is configured.
        """
        if not self._configured:
            raise RuntimeError(
                "No LLM configured. Set CHESTXPERT_LLM_API_URL/"
                "CHESTXPERT_LLM_API_TOKEN or ensure a local transformers "
                "model is available."
            )
        prompt = self._build_v7_prompt(
            patient, diseases, disease_probs, evidence,
            ai_evidence_items, image_summary,
        )
        if self.api_url:
            text = self._call_api(prompt)
        else:
            text = self._call_local(
                prompt, image_path=image_path, contact_sheet_path=contact_sheet_path
            )
        return self._assemble_v7_report(
            text, patient, diseases, disease_probs, ai_evidence_items
        )

    # -- prompt building ---------------------------------------------------

    def _build_v7_prompt(
        self,
        patient: PatientInfo,
        diseases: list[str],
        disease_probs: dict[str, float],
        evidence: dict[str, list[str]],
        ai_evidence_items: list[AIEvidence],
        image_summary: str,
    ) -> str:
        lines: list[str] = []
        lines.append(
            "You are a board-certified radiologist assistant. "
            "Generate a V7 Qwen3-VL All-Disease Radiologist Evidence Fusion "
            "report. Output plain text only, with the exact section headers "
            "shown below."
        )
        lines.append("")
        lines.append("=" * 80)
        lines.append("1. PATIENT / CASE INFORMATION")
        lines.append("=" * 80)
        lines.append(f"- Patient ID: {patient.patient_id}")
        lines.append(f"- Study: {patient.study}")
        lines.append(f"- View: {patient.view}")
        lines.append(f"- Patient age: {patient.patient_age}")
        lines.append(f"- Patient sex: {patient.patient_sex}")
        lines.append(f"- Clinical indication/history: {patient.clinical_indication}")
        lines.append(f"- Comparison study: {patient.comparison_study}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("2. EXAM TECHNIQUE")
        lines.append("=" * 80)
        lines.append(
            f"The examination is a frontal chest radiograph, {patient.view.lower()}. "
            f"{image_summary}"
        )
        lines.append("")
        lines.append("=" * 80)
        lines.append("3. IMAGE QUALITY")
        lines.append("=" * 80)
        lines.append(
            "Assess penetration, exposure, rotation, lung volume, motion artifact, "
            "and positioning adequacy based on the image."
        )
        lines.append("")
        lines.append("=" * 80)
        lines.append("4. FINDINGS")
        lines.append("=" * 80)
        findings_letters = [
            "A. Cardiomediastinal silhouette",
            "B. Lungs",
            "C. Pleura",
            "D. Pulmonary vasculature / interstitial markings",
            "E. Hila / mediastinum",
            "F. Diaphragm / costophrenic angles",
            "G. Bones",
            "H. Soft tissues",
            "I. Support devices / lines / tubes",
        ]
        for letter in findings_letters:
            lines.append("")
            lines.append(letter)
        lines.append("")
        lines.append("=" * 80)
        lines.append("5. AI EVIDENCE SUMMARY")
        lines.append("=" * 80)
        for item in ai_evidence_items:
            lines.append(
                f"- {item.disease}: XRV={item.probability:.4f} "
                f"({item.gradcam_location}). {item.notes}"
            )
        lines.append("")
        lines.append("Other disease probabilities:")
        for d in diseases:
            p = disease_probs.get(d, 0.0)
            lines.append(f"- {d}: {p:.4f}")
        lines.append("")
        lines.append("Retrieved clinical evidence:")
        for d in diseases:
            for e in evidence.get(d, []):
                lines.append(f"- {d}: {e}")
        lines.append("")
        lines.append(
            "Summarize which AI predictions are supported by visible imaging "
            "findings and which are not. Note discrepancies."
        )
        lines.append("")
        lines.append("=" * 80)
        lines.append("6. IMPRESSION")
        lines.append("=" * 80)
        lines.append(
            "Provide a numbered impression (1..N) summarizing the key findings, "
            "supported AI evidence, and recommendations."
        )
        lines.append("")
        lines.append(
            "Output the complete report in the V7 format above. "
            "Ground every statement in the provided image and classifier evidence. "
            "Do not fabricate findings, patient history, or clinical details."
        )
        return "\n".join(lines)

    # -- model calls -------------------------------------------------------

    def _call_local(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        contact_sheet_path: Optional[str] = None,
    ) -> str:
        import torch
        from PIL import Image

        # Collect images for VLM input.
        images: list = []
        seen = set()
        for p in (image_path, contact_sheet_path):
            if p and p not in seen and Path(p).exists():
                seen.add(p)
                images.append(Image.open(p).convert("RGB"))

        if self._is_vlm and images and self._processor is not None:
            # VLM path: use chat template with images (adds required image tokens).
            content = [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": prompt}]
            messages = [
                {
                    "role": "system",
                    "content": ("You are a board-certified radiologist assistant. "
                                "Generate a V7 evidence-fusion radiology report. "
                                "Output plain text only."),
                },
                {"role": "user", "content": content},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=images, return_tensors="pt"
            )
        else:
            # Text-only path.
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            text = prompt

        # Move inputs to the model's device (handles device_map="auto" on CUDA).
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt echo and any VLM chat-template role markers.
        if generated.startswith(text):
            generated = generated[len(text):]
        # Remove role markers that may appear in VLM output.
        for marker in ("\nassistant\n", "assistant\n", "\nassistant ", "\nuser\n",
                       "user\n", "\nuser ", "\nsystem\n", "system\n", "\nsystem "):
            pos = generated.rfind(marker)
            if pos != -1:
                generated = generated[pos + len(marker):].strip()
                break
        else:
            # Last resort: find actual report content.
            for marker in ("Patient ID:", "Study: Chest X-ray",
                           "1. PATIENT / CASE INFORMATION",
                           "QWEN3-VL ALL-DISEASE"):
                pos = generated.rfind(marker)
                if pos != -1:
                    generated = generated[pos:].strip()
                    break
        return generated.strip()

    def _call_api(self, prompt: str) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        r = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("text", data.get("choices", [{}])[0].get("text", ""))

    # -- assembly ----------------------------------------------------------

    def _assemble_v7_report(
        self,
        llm_text: str,
        patient: PatientInfo,
        diseases: list[str],
        disease_probs: dict[str, float],
        ai_evidence_items: list[AIEvidence],
    ) -> V7Report:
        # Strip VLM chat-template role markers before parsing so they
        # do not leak into the rendered report.
        llm_text = self._strip_role_markers(llm_text)

        # Distribute raw LLM text across V7 sections by keyword matching.
        # The LLM reliably ignores V7 section headers, so keyword-based
        # distribution is the primary method for consistent output.
        exam_technique, image_quality, findings, ai_evidence, impression = \
            self._distribute_text_by_section(llm_text, patient)

        full_text = self._render_v7(
            patient, exam_technique, image_quality, findings,
            ai_evidence, impression, ai_evidence_items,
            diseases, disease_probs,
        )

        return V7Report(
            patient=patient,
            exam_technique=exam_technique,
            image_quality=image_quality,
            findings=findings,
            ai_evidence=ai_evidence,
            impression=impression,
            full_text=full_text,
            diseases=diseases,
            confidence=disease_probs,
        )

    @staticmethod
    def _strip_role_markers(text: str) -> str:
        """Remove VLM chat-template role markers and markdown artifacts
        from generated text so they do not leak into the rendered report."""
        cleaned = text
        # Remove role markers.
        markers = [
            "\nassistant\n", "assistant\n", "\nassistant ",
            "\nuser\n", "user\n", "\nuser ",
            "\nsystem\n", "system\n", "\nsystem ",
            "<|assistant|>", "<|user|>", "<|system|>",
        ]
        for marker in markers:
            cleaned = cleaned.replace(marker, "")
        # Remove markdown header markers (####, #####, ###, ##, #).
        import re
        cleaned = re.sub(r"^(#{1,6})\s+", "", cleaned, flags=re.MULTILINE)
        # Remove horizontal rules.
        cleaned = re.sub(r"^[-*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
        # Remove bold/italic markers.
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = cleaned.replace("*", "").replace("_", "")
        return cleaned.strip()

    @staticmethod
    def _distribute_text_by_section(
        llm_text: str, patient: PatientInfo
    ) -> tuple[str, str, dict[str, str], str, str]:
        """Distribute raw LLM text across V7 sections by keyword matching.

        When the LLM does not follow the exact section header format, this
        heuristic assigns paragraphs to the most likely section based on
        finding keywords, ensuring every V7 section is populated.
        """
        lines = [l.strip() for l in llm_text.split("\n") if l.strip()]
        text_lower = llm_text.lower()

        # Finding keyword -> section letter mapping
        finding_keywords = {
            "A": ["cardiomediastinal", "heart", "cardiac", "mediastinum", "aortic"],
            "B": ["lung", "lungs", "pulmonary", "infiltrat", "consolidation", "nodule", "mass"],
            "C": ["pleura", "pleural", "effusion", "costophrenic", "pneumothorax"],
            "D": ["vasculature", "vascular", "interstitial", "edema", "hilar"],
            "E": ["hila", "hilum", "mediastinum", "trachea", "lymphadenopathy"],
            "F": ["diaphragm", "costophrenic", "hemidiaphragm"],
            "G": ["bone", "bones", "rib", "clavicle", "fracture", "spine", "skeletal"],
            "H": ["soft tissue", "soft tissues", "chest wall", "axillary"],
            "I": ["device", "catheter", "tube", "line", "tube", "endotracheal", "ng tube"],
        }

        # Section keyword -> content
        section_content = {
            "2": [],   # exam technique
            "3": [],   # image quality
            "4_A": [], "4_B": [], "4_C": [], "4_D": [], "4_E": [],
            "4_F": [], "4_G": [], "4_H": [], "4_I": [],
            "5": [],   # AI evidence
            "6": [],   # impression
        }

        current_section = None
        for line in lines:
            stripped = line.strip()
            low = stripped.lower()

            # Priority 1: V7 finding subsection headers (4.A. - 4.I., A. - I.).
            m2 = re.match(r"^4?\.?\s*([A-I])\.\s", low)
            if m2 and current_section == "4":
                letter = m2.group(1).upper()
                section_content[f"4_{letter}"] = [stripped]
                current_section = f"4_{letter}"
                continue

            # Priority 2: Finding keywords -> assign to findings A-I.
            # The LLM commonly uses finding-based numbering (e.g. "2. Lungs:"),
            # so keyword matching is more reliable than section headers.
            assigned = False
            for letter, keywords in finding_keywords.items():
                if any(kw in low for kw in keywords):
                    section_content[f"4_{letter}"].append(stripped)
                    current_section = f"4_{letter}"
                    assigned = True
                    break
            if assigned:
                continue

            # Priority 3: V7 main section headers (1-6).
            m = re.match(r"^([1-6])\.\s", low)
            if m:
                num = m.group(1)
                if num in "12356":
                    current_section = num
                elif num == "4":
                    current_section = "4"
                continue

            # Priority 4: Other section keywords (always run when no
            # keyword/section header matched above).
            if any(kw in low for kw in ["technique", "projection", "position", "view", "semi-erect", "pa ", "ap "]):
                current_section = "2"
            elif any(kw in low for kw in ["quality", "penetration", "exposure", "rotation", "inspiration", "volume", "artifact"]):
                current_section = "3"
            elif any(kw in low for kw in ["impression", "recommend", "conclusion", "summary"]):
                current_section = "6"
            elif any(kw in low for kw in ["evidence", "xrv=", "probabilit", "grad-cam", "gradcam"]):
                current_section = "5"
            else:
                current_section = "2"  # default to exam technique

            if current_section in section_content:
                section_content[current_section].append(stripped)

        # Build output strings
        exam_technique = "\n".join(section_content["2"]) or (
            f"The examination is a frontal chest radiograph, "
            f"{patient.view.lower()}."
        )
        image_quality = "\n".join(section_content["3"]) or (
            "Image quality assessment not generated by LLM."
        )

        findings = {}
        finding_titles = {
            "A": "Cardiomediastinal silhouette",
            "B": "Lungs",
            "C": "Pleura",
            "D": "Pulmonary vasculature / interstitial markings",
            "E": "Hila / mediastinum",
            "F": "Diaphragm / costophrenic angles",
            "G": "Bones",
            "H": "Soft tissues",
            "I": "Support devices / lines / tubes",
        }
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
            content = "\n".join(section_content[f"4_{letter}"])
            if content.strip():
                findings[letter] = content.strip()
            else:
                findings[letter] = f"Finding {letter} not generated by LLM."

        ai_evidence = "\n".join(section_content["5"]) or (
            "AI evidence summary not generated by LLM."
        )
        impression = "\n".join(section_content["6"]) or (
            "Impression not generated by LLM."
        )

        return exam_technique, image_quality, findings, ai_evidence, impression

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        # Split on section header markers like "1.", "2.", ..., "6."
        # and subsections "4.A.", "4.B." etc.
        current = None
        buf: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            m = re.match(r"^([1-6])\.\s", stripped)
            if m:
                if current is not None:
                    sections[current] = "\n".join(buf).strip()
                current = m.group(1)
                buf = [stripped]
                continue
            m2 = re.match(r"^4\.([A-I])\.\s", stripped)
            if m2 and current == "4":
                sections[f"4.{m2.group(1)}"] = stripped
                continue
            if current is not None:
                buf.append(line)
        if current is not None:
            sections[current] = "\n".join(buf).strip()
        return sections

    @staticmethod
    def _is_default_or_empty(text: str) -> bool:
        """Return True if the section content is empty or a default fallback."""
        if not text or not text.strip():
            return True
        t = text.strip().lower()
        default_markers = [
            "not generated by llm",
            "not available",
            "n/a",
            "none",
        ]
        return any(m in t for m in default_markers)

    @staticmethod
    def _render_v7(
        patient: PatientInfo,
        exam_technique: str,
        image_quality: str,
        findings: dict[str, str],
        ai_evidence: str,
        impression: str,
        ai_evidence_items: list[AIEvidence],
        diseases: list[str],
        disease_probs: dict[str, float],
    ) -> str:
        lines: list[str] = []
        lines.append(
            "QWEN3-VL ALL-DISEASE RADIOLOGIST EVIDENCE FUSION"
        )
        lines.append("=" * 80)
        lines.append("")

        # ---- Section 1: Patient / Case Information (always rendered) ----
        lines.append("===============================================================================")
        lines.append("1. PATIENT / CASE INFORMATION")
        lines.append("===============================================================================")
        lines.append("")
        lines.append(f"- Patient ID: {patient.patient_id}")
        lines.append(f"- Study: {patient.study}")
        lines.append(f"- View: {patient.view}")
        lines.append(f"- Patient age: {patient.patient_age}")
        lines.append(f"- Patient sex: {patient.patient_sex}")
        lines.append(f"- Clinical indication/history: {patient.clinical_indication}")
        lines.append(f"- Comparison study: {patient.comparison_study}")
        lines.append("")

        # ---- Section 2: Exam Technique ----
        if not V7LLMClient._is_default_or_empty(exam_technique):
            lines.append("===============================================================================")
            lines.append("2. EXAM TECHNIQUE")
            lines.append("===============================================================================")
            lines.append("")
            lines.append(exam_technique)
            lines.append("")

        # ---- Section 3: Image Quality ----
        if not V7LLMClient._is_default_or_empty(image_quality):
            lines.append("===============================================================================")
            lines.append("3. IMAGE QUALITY")
            lines.append("===============================================================================")
            lines.append("")
            lines.append(image_quality)
            lines.append("")

        # ---- Section 4: Findings (A-I) ----
        has_any_finding = False
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
            content = findings.get(letter, "")
            if not V7LLMClient._is_default_or_empty(content):
                if not has_any_finding:
                    lines.append("===============================================================================")
                    lines.append("4. FINDINGS")
                    lines.append("===============================================================================")
                    lines.append("")
                    has_any_finding = True
                finding_titles = {
                    "A": "Cardiomediastinal silhouette",
                    "B": "Lungs",
                    "C": "Pleura",
                    "D": "Pulmonary vasculature / interstitial markings",
                    "E": "Hila / mediastinum",
                    "F": "Diaphragm / costophrenic angles",
                    "G": "Bones",
                    "H": "Soft tissues",
                    "I": "Support devices / lines / tubes",
                }
                lines.append("")
                lines.append(f"{letter}. {finding_titles[letter]}")
                lines.append(content)
        if has_any_finding:
            lines.append("")

        # ---- Section 5: AI Evidence Summary ----
        has_ai_evidence = not V7LLMClient._is_default_or_empty(ai_evidence)
        if has_ai_evidence or ai_evidence_items:
            lines.append("===============================================================================")
            lines.append("5. AI EVIDENCE SUMMARY")
            lines.append("===============================================================================")
            lines.append("")
            if not V7LLMClient._is_default_or_empty(ai_evidence):
                lines.append(ai_evidence)
                lines.append("")
            # Always include structured AI evidence items from the classifier,
            # so section 5 is never empty even if the LLM omits it.
            if ai_evidence_items:
                lines.append("Classifier evidence (XRV scores):")
                for item in ai_evidence_items:
                    status = "SUPPORTED" if item.supported else "NOT SUPPORTED"
                    lines.append(
                        f"- {item.disease}: XRV={item.probability:.4f} "
                        f"[{status}] Grad-CAM: {item.gradcam_location or 'n/a'}. "
                        f"{item.notes}"
                    )
                lines.append("")
            lines.append("Other disease probabilities:")
            for d in diseases:
                p = disease_probs.get(d, 0.0)
                lines.append(f"- {d}: {p:.4f}")
            lines.append("")

        # ---- Section 6: Impression ----
        if not V7LLMClient._is_default_or_empty(impression):
            lines.append("===============================================================================")
            lines.append("6. IMPRESSION")
            lines.append("===============================================================================")
            lines.append("")
            lines.append(impression)
            lines.append("")

        return "\n".join(lines)


def build_kg_from_reports(reports: list[tuple[str, str]]) -> ClinicalKnowledgeGraph:
    """Build a ClinicalKnowledgeGraph from a list of (report_id, text)."""
    kg = ClinicalKnowledgeGraph()
    for rid, text in reports:
        kg.add_report(rid, text)
    return kg
