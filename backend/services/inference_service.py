"""Inference service wrapping the existing Chest Xpert modules.

Reuses preprocessing, modality validation, the selected CNN ensemble, Grad-CAM,
and report generation from the chest_xpert package. Runs expensive operations
in background jobs with bounded concurrency.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

# Existing Chest Xpert modules (imported lazily to avoid torch overhead at import
# time when only health checks are needed).
from backend.config import (
    MODEL_CONFIG,
    MODALITY_CONFIG,
    STORAGE_CONFIG,
)
from backend.services.resource_manager import MONITOR

# Deferred imports (torch-heavy):
#   chest_xpert.config as CXConfig
#   chest_xpert.preprocessing import validate_image, preprocess
#   chest_xpert.modality import ModalityValidator
#   chest_xpert.inference import InferenceEngine
#   chest_xpert.label_mapping import map_outputs_to_labels, scores_dict
#   chest_xpert.explainability import GradCAM
#   chest_xpert.report import ReportData, ExamDetails, generate_html, generate_pdf
#   chest_xpert.verification import VerificationEngine, LLMModelClient


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    analysis_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    image_path: Optional[str] = None
    grad_cam_path: Optional[str] = None
    report_path: Optional[str] = None


class InferenceService:
    """Singleton-style service that owns the model lifecycle and job queue.

    The classifier model is loaded once at startup (or on first use) and
    reused across requests. GPU-heavy jobs are serialized via an asyncio
    semaphore bounded by MAX_CONCURRENT_JOBS.
    """

    def __init__(self) -> None:
        self._engine: Optional[object] = None
        self._engine_device: Optional[str] = None
        self._modality_validator: Optional[object] = None
        self._verification_engine: Optional[object] = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(MODEL_CONFIG.max_concurrent_jobs)
        self._jobs: dict[str, JobRecord] = {}
        self._model_loaded = False
        self._model_load_error: Optional[str] = None
        self._load_snapshot: Optional[dict] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    async def ensure_model_loaded(self) -> tuple[bool, Optional[str]]:
        """Load the selected CNN once. Returns (ok, error).

        Falls back to CPU if CUDA is out of memory (common on the shared V100).
        """
        if self._model_loaded:
            return True, None

        async with self._lock:
            if self._model_loaded:
                return True, None

            try:
                from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG as CX_MODEL_CONFIG
                from chest_xpert.model_selection import selection
                if selection().get('cnn', {}).get('architecture') == 'densenet201_v6_v12_tta':
                    from chest_xpert.selected_ensemble import SelectedEnsembleEngine as InferenceEngine
                else:
                    from chest_xpert.checkpoint_engine import CheckpointEngine as InferenceEngine

                device = MODEL_CONFIG.device if MODEL_CONFIG.device != "auto" else "cuda:0"

                def _load(dev: str) -> InferenceEngine:
                    if InferenceEngine.__name__ == 'SelectedEnsembleEngine':
                        engine = InferenceEngine(device=dev)
                    else:
                        engine = InferenceEngine(model_name=CX_MODEL_CONFIG.name, device=dev)
                    _ = engine.model  # force load
                    return engine

                t0 = time.perf_counter()
                before = MONITOR.snapshot("before_load")

                engine = await asyncio.to_thread(_load, device)

                load_ms = (time.perf_counter() - t0) * 1000.0
                after = MONITOR.snapshot("after_load")

                self._engine = engine
                self._engine_device = device
                self._model_loaded = True
                self._model_load_error = None
                self._load_snapshot = {
                    "before": before,
                    "after": after,
                    "load_time_ms": round(load_ms, 2),
                }
                return True, None
            except Exception as exc:  # noqa: BLE001
                self._model_load_error = str(exc)
                return False, str(exc)

    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def engine(self) -> Optional[object]:
        return self._engine

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------
    def create_job(self, analysis_id: str, image_path: str) -> JobRecord:
        job = JobRecord(analysis_id=analysis_id, image_path=image_path)
        self._jobs[analysis_id] = job
        return job

    def get_job(self, analysis_id: str) -> Optional[JobRecord]:
        return self._jobs.get(analysis_id)

    def list_jobs(self) -> list[JobRecord]:
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Main analysis entry point (called by background worker)
    # ------------------------------------------------------------------
    async def run_analysis(
        self,
        analysis_id: str,
        image_path: str,
        request_data: dict,
        generate_grad_cam: bool = False,
    ) -> None:
        """Run the full 5-stage pipeline for one analysis.

        Executes within the concurrency semaphore so GPU-heavy jobs are
        serialized. The model is loaded before entering the worker thread
        so no async calls are made from a thread context.
        """
        job = self.get_job(analysis_id)
        if job is None:
            return

        # Ensure the model is loaded before dispatching to the worker thread.
        ok, err = await self.ensure_model_loaded()
        if not ok:
            job.status = JobStatus.FAILED
            job.error = f"Model load failed: {err}"
            job.completed_at = time.time()
            return

        async with self._semaphore:
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            try:
                result = await asyncio.to_thread(
                    self._run_pipeline_sync,
                    analysis_id,
                    image_path,
                    request_data,
                    generate_grad_cam,
                )
                job.result = result
                if result.get("status") == "rejected":
                    job.status = JobStatus.FAILED
                    job.error = result["reason"]
                else:
                    job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
            except Exception as exc:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.time()
                # Release GPU resources on failure (defensive, wrapped).
                try:
                    MONITOR.cleanup()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Synchronous pipeline (runs in a worker thread)
    # ------------------------------------------------------------------
    def _run_pipeline_sync(
        self,
        analysis_id: str,
        image_path: str,
        request_data: dict,
        generate_grad_cam: bool,
    ) -> dict:
        from chest_xpert.config import (
            ALLOWED_EXTENSIONS,
            MAX_IMAGE_MB,
            DEVICE_CONFIG,
            DISCLAIMER,
            LABELS_18,
            MODEL_CONFIG as CX_MODEL_CONFIG,
        )
        from chest_xpert.label_mapping import map_outputs_to_labels, scores_dict
        from chest_xpert.modality import modality_validator
        from chest_xpert.preprocessing import validate_image, preprocess

        path = Path(image_path)
        view_paths = [path] + [Path(p) for p in request_data.get("additional_image_paths", [])]
        if len(view_paths) > 5 or len({p.resolve() for p in view_paths}) != len(view_paths):
            raise ValueError("Provide one to five distinct original images from the same study")

        # Ensure consistent CUDA device across all operations (multi-GPU safety).
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.set_device(torch.device(self._engine_device or "cuda:0"))

        modality_total_ms = 0.0
        for view_path in view_paths:
            # ---------- Stage 0: file validation ----------
            validation = validate_image(view_path)
            if not validation.ok:
                return {
                    "analysis_id": analysis_id,
                    "status": "rejected",
                    "reason": validation.reason,
                    "modality": {
                        "decision": "technical_failure",
                        "reason": validation.reason,
                        "model": None,
                        "latency_ms": None,
                    },
                    "labels": [],
                    "disclaimer": DISCLAIMER,
                }

            # ---------- Stage 0: modality validation ----------
            modality_decision = "accepted"
            modality_reason = "File validation passed; modality check skipped."
            modality_model = None
            modality_ms = None
            modality_accepted = True

            if MODALITY_CONFIG.enabled:
                validator = modality_validator
                t0 = time.perf_counter()
                mod_result = validator.validate(view_path)
                modality_ms = (time.perf_counter() - t0) * 1000.0
                modality_total_ms += modality_ms
                modality_decision = mod_result.decision.value
                modality_reason = mod_result.reason
                modality_model = mod_result.model_id
                modality_accepted = mod_result.is_accepted

                if not mod_result.is_accepted:
                    return {
                        "analysis_id": analysis_id,
                        "status": "rejected",
                        "reason": ("The uploaded image is not a chest X-ray. Please upload a valid chest X-ray." if mod_result.is_rejected else f"Chest X-ray validation could not confirm this image: {modality_reason}"),
                        "modality": {
                            "decision": modality_decision,
                            "reason": modality_reason,
                            "model": modality_model,
                            "latency_ms": round(modality_ms, 2),
                        },
                        "labels": [],
                        "disclaimer": DISCLAIMER,
                    }

        modality_ms = modality_total_ms

        # ---------- Stage A: classifier inference ----------
        # Model is already loaded by run_analysis before entering this thread.
        if self._engine is None:
            raise RuntimeError("Inference engine not loaded.")

        engine = self._engine
        device = self._engine_device or "cpu"

        t0 = time.perf_counter()
        from chest_xpert.checkpoint_engine import prepare_checkpoint_image
        from types import SimpleNamespace
        selected_ensemble = hasattr(engine, 'predict_pair')
        if selected_ensemble:
            prepared_views = [SimpleNamespace(pair=engine.prepare(p)) for p in view_paths]
            predictions = [engine.predict_pair(prepared.pair) for prepared in prepared_views]
        else:
            prepared_views = [SimpleNamespace(tensor=prepare_checkpoint_image(p,engine.resolution,engine.pixel_scale,engine.crop).unsqueeze(0).to(device)) for p in view_paths]
            predictions = [engine.predict(prepared.tensor) for prepared in prepared_views]
        per_view_scores = [prediction.raw.copy() for prediction in predictions]
        pp = prepared_views[0]
        result = predictions[0]
        result.raw = {label: sum(scores[label] for scores in per_view_scores)/len(per_view_scores) for label in result.raw}
        result.inference_only_ms = sum(prediction.inference_only_ms for prediction in predictions)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        from chest_xpert.label_mapping import LabelMapping
        from chest_xpert.report_quality import evidence_warnings, validate_clinical_sections, report_warnings, report_context_warnings
        quality_warnings = evidence_warnings(result.raw)
        mappings = [LabelMapping(label, True, label, score) for label, score in result.raw.items()]
        label_list = []
        for m in mappings:
            sev = "not assessed"
            label_list.append(
                {
                    "label": m.requested_label,
                    "supported": m.supported,
                    "score": m.score,
                    "severity": sev,
                }
            )

        generate_grad_cam = True
        # ---------- Grad-CAM for top findings + contact sheet ----------
        grad_cam_path = None
        grad_cam_ms = None
        cam_results: list = []
        supported = []
        # A contact sheet is required evidence even if a legacy caller disables display.
        supported = sorted(
            [m for m in mappings if m.supported and m.score is not None],
            key=lambda m: m.score or 0,
            reverse=True,
        )[:6]  # Top 6 findings only — reduces latency
        if supported:
            t0 = time.perf_counter()
            import numpy as np

            from chest_xpert.explainability import (
                GradCAM,
                generate_contact_sheet,
                save_contact_sheet,
            )

            sheet_name = f"{analysis_id}_gradcam_contact_sheet.png"
            sheet_path = STORAGE_CONFIG.temp_dir / sheet_name
            if selected_ensemble:
                engine.contact_sheet([prepared.pair for prepared in prepared_views],result.raw,sheet_path)
            else:
                cam = GradCAM(engine.model, target_layer_name="encoder")
                from dataclasses import replace
                displays=[]
                for view_index,prepared in enumerate(prepared_views):
                    disp = prepared.tensor.detach().cpu().numpy()[0, 0]
                    disp = np.clip((disp / engine.pixel_scale + 1) / 2, 0, 1)
                    displays.append(disp)
                    for m in supported:
                        grad = cam.generate(input_tensor=prepared.tensor,target_label=m.model_output,model_outputs=result.model_outputs,original_image=disp)
                        if len(prepared_views)>1:
                            grad=replace(grad,target_label=f"View {view_index+1}: {grad.target_label}")
                        cam_results.append(grad)
                cam.close()
                sheet = generate_contact_sheet(cam_results, original_image=displays[0],
                    cols=6 if len(prepared_views)>1 else 3,
                    rows=len(prepared_views) if len(prepared_views)>1 else 2, cell_size=224)
                save_contact_sheet(sheet, str(sheet_path))
            grad_cam_ms = (time.perf_counter() - t0) * 1000.0
            grad_cam_path = str(sheet_path)


        if not grad_cam_path or not Path(grad_cam_path).is_file():
            raise RuntimeError("A Grad-CAM contact sheet is required before VLM report generation.")

        # ---------- Stage B: Qwen V7 report generation ----------
        verification_info = {
            "available": False,
            "model": None,
            "latency_ms": None,
            "error": None,
            "agreement": None,
        }
        report_findings = None
        v7_report_text = None

        from chest_xpert.verification import VerificationEngine

        from PIL import Image
        converted_paths=[]
        for i,source_path in enumerate(view_paths):
            suffix="original" if i==0 else f"original_view_{i+1}"
            converted = STORAGE_CONFIG.temp_dir / f"{analysis_id}_{suffix}.png"
            if source_path.suffix.lower() in {".dcm", ".dicom"}:
                from chest_xpert.preprocessing import load_image
                import numpy as np
                arr = load_image(source_path)
                Image.fromarray(np.uint8(np.clip((arr + 1024) / 2048, 0, 1) * 255)).save(converted)
            else:
                with Image.open(source_path) as image:
                    image.convert("RGB").save(converted)
            converted_paths.append(str(converted))
        image_path = converted_paths[0] if len(converted_paths)==1 else converted_paths
        ve = self._verification_engine or VerificationEngine()
        self._verification_engine = ve
        if ve.client.backend != "transformers_local":
            raise RuntimeError("This workflow requires the local CUDA vision backend so both images are supplied. Set CHESTXPERT_LLM_BACKEND=transformers_local.")
        model_doc = (
            "Selected DenseNet201 ensemble with 14 finding outputs. "
            "Scores are uncalibrated CNN outputs, not calibrated probabilities."
        )

        if engine.crop == "center":
            model_doc += " Grad-CAM uses the centered square CNN crop; the original vision image is uncropped. Do not infer heatmap localization outside the crop."

        patient_info = {
            "patient_id": request_data.get("patient_id") or "NOT PROVIDED",
            "original_image_count": len(view_paths),
            "study": request_data.get("modality", "Chest X-ray"),
            "report_sections": __import__("backend.services.review_store", fromlist=["template"]).template(request_data.get("template_id", "standard"))["sections"],
            "view": request_data.get("view") or "NOT PROVIDED",
            "technique": ("Chest radiograph, " + request_data["view"] + " projection.") if request_data.get("view") else "NOT PROVIDED",
            "patient_age": "NOT PROVIDED",
            "patient_sex": "NOT PROVIDED",
            "clinical_indication": request_data.get("clinical_indication") or "NOT PROVIDED",
            "comparison_study": request_data.get("comparison_study") or "NOT PROVIDED",
        }

        from backend.services import review_store
        template = review_store.template(request_data.get("template_id", "standard"))
        model_doc += " " + " ".join(quality_warnings)
        model_doc += " Return ONLY a JSON object with key sections: a list of objects with heading and text. Use exactly these section headings in order: " + __import__("json").dumps(template["sections"]) + ". Assess the original image; scores are uncalibrated supporting signals, never disease probability or severity. Do not invent unavailable patient details."
        t0 = time.perf_counter()
        vr = ve.generate_v7_report(
            mappings=mappings,
            model_doc=model_doc,
            patient_info=patient_info,
            image_path=image_path,
            contact_sheet_path=grad_cam_path,
            self_correct=False,
        )
        verification_ms = (time.perf_counter() - t0) * 1000.0
        verification_info["available"] = vr.available
        verification_info["model"] = vr.model_id or None
        verification_info["latency_ms"] = round(vr.latency_ms, 2)
        verification_info["error"] = vr.error
        v7_report_text = vr.v7_report
        if not vr.available or not v7_report_text:
            raise RuntimeError(vr.error or "Vision model did not produce a report.")

        import json
        from chest_xpert.report_quality import parse_generated_sections
        try:
            sections, schema_normalized = parse_generated_sections(v7_report_text, template["sections"])
        except (ValueError, json.JSONDecodeError):
            # Keep all original images, CNN scores, and contact-sheet evidence
            # when retrying an invalid JSON draft. Never invent missing text.
            retry_doc = model_doc + (
                " Return one syntactically valid JSON object only. Escape quotes "
                "and newlines inside text strings. Close every object and array; "
                "do not include markdown or commentary."
            )
            vr = ve.generate_v7_report(
                mappings=mappings,
                model_doc=retry_doc,
                patient_info=patient_info,
                image_path=image_path,
                contact_sheet_path=grad_cam_path,
                self_correct=False,
            )
            if not vr.available or not vr.v7_report:
                raise RuntimeError(vr.error or "Vision model did not produce a valid report")
            v7_report_text = vr.v7_report
            sections, schema_normalized = parse_generated_sections(v7_report_text, template["sections"])
            verification_ms = (time.perf_counter() - t0) * 1000.0
            verification_info.update(available=True, model=vr.model_id or None,
                                     latency_ms=round(verification_ms, 2), error=None)
            quality_warnings.append("The first draft had invalid JSON and was regenerated for review.")
        if schema_normalized:
            quality_warnings.append("Generated JSON layout was normalized; clinical text was preserved for review.")
        # Metadata sections must come from supplied metadata, never model guesses.
        for section in sections:
            key = section["heading"].strip().lower()
            if key in {"clinical indication", "clinical history", "indication", "history"}:
                section["text"] = request_data.get("clinical_indication") or "NOT PROVIDED"
            elif key in {"comparison", "comparison study"}:
                section["text"] = request_data.get("comparison_study") or "NOT PROVIDED"
            elif key in {"technique", "exam technique"}:
                section["text"] = patient_info["technique"]
        quality_warnings.extend(report_warnings(sections))
        quality_warnings.extend(report_context_warnings(sections, request_data.get("comparison_study")))
        review_store.write(analysis_id, {"analysis_id": analysis_id, "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "patient_id": request_data.get("patient_id"), "sections": sections, "template": template, "signature": None, "revision": 1, "labels": label_list, "quality_warnings": quality_warnings, "model_provenance": {"cnn": engine.model_name, "cnn_fingerprint": engine.fingerprint, "vision_model": ve.client.model_id, "evidence_mode": "cnn_evidence", "vision_inputs": getattr(ve.client, "_last_vision_inputs", None), "cnn_scores": result.raw, "cnn_scores_by_view": per_view_scores, "score_aggregation": "mean across supplied views", "adapter_name": Path(ve.client.lora_adapter_path).name if ve.client.lora_adapter_path else None}})

        # ---------- Stage C/D: report data ----------
        from chest_xpert.report import ExamDetails, ReportData, generate_html, generate_pdf

        rd = ReportData()
        rd.exam = ExamDetails(
            patient_id=request_data.get("patient_id"),
            exam_date=request_data.get("exam_date"),
            modality=request_data.get("modality", "Chest X-ray"),
            view=request_data.get("view"),
            institution=request_data.get("institution"),
            referring_physician=request_data.get("referring_physician"),
            device=device,
        )
        rd.model_name = result.model_name
        rd.model_version = "pre-trained weights (torchxrayvision)"
        rd.device = device
        rd.latency_ms = round(inference_ms, 2)
        rd.inference_only_ms = round(result.inference_only_ms, 2)
        rd.modality_ms = round(modality_ms, 2) if modality_ms else None
        rd.modality_accepted = modality_accepted
        rd.modality_decision = modality_decision
        rd.modality_reason = modality_reason
        rd.modality_model = modality_model
        rd.mappings = mappings
        rd.unsupported_labels = result.unsupported_labels
        rd.heatmap_path = grad_cam_path
        rd.heatmap_label = (
            supported[0].requested_label if (generate_grad_cam and supported) else None
        )
        rd.contact_sheet_path = grad_cam_path
        rd.verification_available = verification_info["available"]
        rd.verification_error = verification_info["error"]
        rd.verification_result = vr
        rd.verification_ms = verification_ms
        rd.impression = request_data.get("impression", "")
        rd.reviewer_name = request_data.get("reviewer_name")
        rd.review_status = request_data.get("review_status", "Pending")
        rd.notes = request_data.get("notes", "")
        rd.v7_report = v7_report_text

        total_ms = inference_ms + (modality_ms or 0) + (grad_cam_ms or 0) + verification_ms

        # Persist the same structured report used by review and export.
        html = review_store.render(review_store.read(analysis_id))
        report_path = STORAGE_CONFIG.reports_dir / f"{analysis_id}.html"
        report_path.write_text(html, encoding="utf-8")

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "modality": {
                "decision": modality_decision,
                "reason": modality_reason,
                "model": modality_model,
                "latency_ms": round(modality_ms, 2) if modality_ms else None,
            },
            "labels": label_list,
            "inference_latency_ms": round(inference_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "report_available": True,
            "grad_cam_available": grad_cam_path is not None,
            "grad_cam_path": grad_cam_path,
            "grad_cam_ms": round(grad_cam_ms, 2) if grad_cam_ms else None,
            "verification": verification_info,
            "report_findings": report_findings,
            "report_html_path": str(report_path),
            "v7_report": v7_report_text,
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _severity(score) -> str:
        return "not assessed"


    def generate_report_export(
        self,
        analysis_id: str,
        fmt: str = "pdf",
        impression: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        review_status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> tuple[str, str]:  # (format, file_path)
        from backend.services import review_store
        if fmt not in {"pdf", "html"}:
            raise ValueError("Supported report formats are PDF and HTML.")
        data = review_store.read(analysis_id)
        html = review_store.render(data)
        out_path = STORAGE_CONFIG.reports_dir / f"{analysis_id}.{fmt}"
        if fmt == "pdf":
            from weasyprint import HTML
            HTML(string=html).write_pdf(str(out_path))
        else:
            out_path.write_text(html, encoding="utf-8")
        return fmt, str(out_path)


inference_service = InferenceService()
