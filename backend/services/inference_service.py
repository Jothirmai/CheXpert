"""Inference service wrapping the existing Chest Xpert modules.

Reuses preprocessing, modality validation, DenseNet-121 inference, Grad-CAM,
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
        """Load the DenseNet-121 model once. Returns (ok, error).

        Falls back to CPU if CUDA is out of memory (common on the shared V100).
        """
        if self._model_loaded or self._model_load_error is not None:
            return self._model_loaded, self._model_load_error

        async with self._lock:
            if self._model_loaded or self._model_load_error is not None:
                return self._model_loaded, self._model_load_error

            try:
                from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG as CX_MODEL_CONFIG
                from chest_xpert.inference import InferenceEngine

                device = DEVICE_CONFIG.resolve() if MODEL_CONFIG.device == "auto" else MODEL_CONFIG.device

                def _load(dev: str) -> InferenceEngine:
                    engine = InferenceEngine(model_name=CX_MODEL_CONFIG.name, device=dev)
                    _ = engine.model  # force load
                    return engine

                t0 = time.perf_counter()
                before = MONITOR.snapshot("before_load")

                try:
                    engine = _load(device)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and device != "cpu":
                        # Fall back to CPU.
                        import torch as _torch
                        _torch.cuda.empty_cache()
                        device = "cpu"
                        engine = _load("cpu")
                    else:
                        raise

                load_ms = (time.perf_counter() - t0) * 1000.0
                after = MONITOR.snapshot("after_load")

                self._engine = engine
                self._engine_device = device
                self._model_loaded = True
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
                raise

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
        from chest_xpert.modality import ModalityValidator
        from chest_xpert.preprocessing import validate_image, preprocess

        path = Path(image_path)

        # ---------- Stage 0: file validation ----------
        validation = validate_image(path)
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
            validator = ModalityValidator()
            t0 = time.perf_counter()
            mod_result = validator.validate(path)
            modality_ms = (time.perf_counter() - t0) * 1000.0
            modality_decision = mod_result.decision.value
            modality_reason = mod_result.reason
            modality_model = mod_result.model_id
            modality_accepted = mod_result.is_accepted

            if mod_result.is_rejected or mod_result.is_uncertain:
                return {
                    "analysis_id": analysis_id,
                    "status": "rejected",
                    "reason": f"Modality {modality_decision}: {modality_reason}",
                    "modality": {
                        "decision": modality_decision,
                        "reason": modality_reason,
                        "model": modality_model,
                        "latency_ms": round(modality_ms, 2),
                    },
                    "labels": [],
                    "disclaimer": DISCLAIMER,
                }

        # ---------- Stage A: classifier inference ----------
        # Model is already loaded by run_analysis before entering this thread.
        if self._engine is None:
            raise RuntimeError("Inference engine not loaded.")

        engine = self._engine
        device = self._engine_device or "cpu"

        t0 = time.perf_counter()
        pp = preprocess(path, device, resolution=CX_MODEL_CONFIG.input_resolution)
        result = engine.predict(pp.tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        mappings = map_outputs_to_labels(result.raw, result.model_name)
        label_list = []
        for m in mappings:
            sev = self._severity(m.score)
            label_list.append(
                {
                    "label": m.requested_label,
                    "supported": m.supported,
                    "severity": sev,
                }
            )

        # ---------- Grad-CAM for ALL supported findings + contact sheet ----------
        grad_cam_path = None
        grad_cam_ms = None
        cam_results: list = []
        supported = []
        if generate_grad_cam:
            supported = [m for m in mappings if m.supported]
            if supported:
                t0 = time.perf_counter()
                import numpy as np

                from chest_xpert.explainability import (
                    GradCAM,
                    generate_contact_sheet,
                    save_contact_sheet,
                )

                cam = GradCAM(engine.model, target_layer_name="features.norm5")
                disp = pp.tensor.detach().cpu().numpy()[0, 0]
                disp = np.clip((disp + 1024) / 2048, 0, 1)

                for m in supported:
                    grad = cam.generate(
                        input_tensor=pp.tensor,
                        target_label=m.model_output,
                        model_outputs=result.model_outputs,
                        original_image=disp,
                    )
                    cam_results.append(grad)
                cam.close()

                grad_cam_ms = (time.perf_counter() - t0) * 1000.0

                # Build contact sheet (6 cols x 3 rows = 18 cells).
                sheet = generate_contact_sheet(
                    cam_results,
                    original_image=disp,
                    cols=6,
                    rows=3,
                    cell_size=224,
                )
                sheet_name = f"{analysis_id}_gradcam_contact_sheet.png"
                sheet_path = STORAGE_CONFIG.temp_dir / sheet_name
                save_contact_sheet(sheet, str(sheet_path))
                grad_cam_path = str(sheet_path)

        # ---------- Stage B: Qwen report generation (required) ----------
        verification_info = {
            "available": False,
            "model": None,
            "latency_ms": None,
            "error": None,
            "agreement": None,
        }
        report_findings = None

        from chest_xpert.verification import VerificationEngine

        ve = VerificationEngine()
        model_doc = (
            "torchxrayvision DenseNet-121 (densenet121-res224-all), "
            "pre-trained weights by other researchers. Scores are raw model "
            "outputs, not calibrated probabilities."
        )
        t0 = time.perf_counter()
        vr = ve.verify(
            mappings=mappings,
            model_doc=model_doc,
            image_path=image_path,
            contact_sheet_path=grad_cam_path,
        )
        verification_ms = (time.perf_counter() - t0) * 1000.0
        verification_info["available"] = vr.available
        verification_info["model"] = vr.model_id or None
        verification_info["latency_ms"] = round(vr.latency_ms, 2)
        verification_info["error"] = vr.error
        n_cons = sum(1 for it in vr.items if it.assessment == "consistent")
        n_incon = sum(1 for it in vr.items if it.assessment == "inconsistent")
        n_unc = sum(1 for it in vr.items if it.assessment == "uncertain")
        verification_info["agreement"] = (
            f"{n_cons} consistent · {n_incon} inconsistent · {n_unc} uncertain"
        )
        if vr.report_findings:
            report_findings = [
                {
                    "pathology": f.pathology,
                    "status": f.status,
                    "evidence_basis": f.evidence_basis,
                    "recommended_wording": f.recommended_wording,
                }
                for f in vr.report_findings
            ]

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

        total_ms = inference_ms + (modality_ms or 0) + (grad_cam_ms or 0) + verification_ms

        # Persist HTML report.
        html = generate_html(rd)
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
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _severity(score) -> str:
        if score is None:
            return "not assessed"
        try:
            score = float(score)
        except (TypeError, ValueError):
            return "not assessed"
        if score >= 0.7:
            return "high"
        if score >= 0.5:
            return "moderate"
        if score >= 0.3:
            return "low"
        return "minimal"

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------
    def generate_report_export(
        self,
        analysis_id: str,
        fmt: str = "pdf",
        impression: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        review_status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> tuple[str, str]:  # (format, file_path)
        from chest_xpert.report import ReportData, generate_html, generate_pdf

        job = self.get_job(analysis_id)
        if job is None or job.result is None:
            raise FileNotFoundError(f"Analysis {analysis_id} not found or not completed.")

        html_path = STORAGE_CONFIG.reports_dir / f"{analysis_id}.html"
        if not html_path.exists():
            raise FileNotFoundError(f"Report HTML not found for {analysis_id}.")

        # Rebuild ReportData from stored result (minimal fields for export).
        rd = ReportData()
        rd.model_name = MODEL_CONFIG.model_name
        rd.model_version = "pre-trained weights (torchxrayvision)"
        rd.device = job.result.get("modality", {}).get("model", "cpu") or "cpu"
        rd.latency_ms = job.result.get("inference_latency_ms", 0)
        rd.modality_decision = job.result.get("modality", {}).get("decision")
        rd.modality_reason = job.result.get("modality", {}).get("reason")
        rd.modality_model = job.result.get("modality", {}).get("model")
        rd.modality_ms = job.result.get("modality", {}).get("latency_ms")
        rd.impression = impression or ""
        rd.reviewer_name = reviewer_name
        rd.review_status = review_status or "Pending"
        rd.notes = notes or ""

        out_path = STORAGE_CONFIG.reports_dir / f"{analysis_id}.{fmt}"
        if fmt == "pdf":
            generate_pdf(rd, out_path)
        else:
            out_path.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

        return fmt, str(out_path)


inference_service = InferenceService()
