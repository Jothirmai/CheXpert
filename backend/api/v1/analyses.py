"""Analysis endpoints: create, get, list."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Security, UploadFile, status

from backend.config import STORAGE_CONFIG
from backend.security.auth import get_api_key
from backend.services.inference_service import InferenceService, JobStatus
from backend.services.job_manager import JobManager
from backend.schemas import (
    AnalysisCreateRequest,
    AnalysisResponse,
    ErrorResponse,
    LabelScore,
    ModalityInfo,
    VerificationInfo,
)

router = APIRouter(tags=["analyses"])

# Module-level singletons (wired in main.py lifespan).
_inference_service: Optional[InferenceService] = None
_job_manager: Optional[JobManager] = None


def init_routers(service: InferenceService, job_manager: JobManager) -> None:
    global _inference_service, _job_manager
    _inference_service = service
    _job_manager = job_manager


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_analysis(
    file: UploadFile = File(...),
    additional_files: list[UploadFile] | None = File(None),
    template_id: str = Form("standard"),
    clinical_indication: Optional[str] = Form(None),
    comparison_study: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    exam_date: Optional[str] = Form(None),
    modality: Optional[str] = Form("Chest X-ray"),
    view: Optional[str] = Form(None),
    institution: Optional[str] = Form(None),
    referring_physician: Optional[str] = Form(None),
    generate_grad_cam: bool = Form(True),
    api_key: str | None = Security(get_api_key),
) -> AnalysisResponse:
    """Upload a chest X-ray and queue it for analysis.

    Returns an analysis_id immediately. Poll GET /analyses/{id} for status.
    """
    if _inference_service is None or _job_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend services not initialized.",
        )

    from backend.services import review_store
    try:
        review_store.template(template_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(400, "Selected template does not exist")

    # Validate extension client-side hint + server-side check.
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    # Read and size-check.
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > STORAGE_CONFIG.max_upload_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB > {STORAGE_CONFIG.max_upload_mb} MB)",
        )

    additional_files = additional_files or []
    if len(additional_files) > 4:
        raise HTTPException(400, "Upload at most five original images from the same study")
    extras=[]
    total_size=len(contents)
    for extra in additional_files:
        suffix=Path(extra.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, "Unsupported file type in additional study image")
        body=await extra.read(STORAGE_CONFIG.max_upload_mb*1024*1024+1)
        total_size+=len(body)
        if total_size>STORAGE_CONFIG.max_upload_mb*1024*1024:
            raise HTTPException(413, "Combined study images exceed the upload size limit")
        extras.append((suffix,body))

    analysis_id = str(uuid.uuid4())
    # Save to a temp path with a safe, unpredictable name.
    safe_name = f"{analysis_id}{ext}"
    tmp_path = STORAGE_CONFIG.temp_dir / safe_name
    tmp_path.write_bytes(contents)

    additional_paths=[]
    for index,(suffix,body) in enumerate(extras,start=2):
        extra_path=STORAGE_CONFIG.temp_dir/f"{analysis_id}_view_{index}{suffix}"
        extra_path.write_bytes(body)
        additional_paths.append(str(extra_path))

    request_data = {
        "additional_image_paths": additional_paths,
        "template_id": template_id,
        "clinical_indication": clinical_indication,
        "comparison_study": comparison_study,
        "patient_id": patient_id,
        "exam_date": exam_date,
        "modality": modality,
        "view": view,
        "institution": institution,
        "referring_physician": referring_physician,
        "impression": None,
        "reviewer_name": None,
        "review_status": "Pending",
        "notes": None,
    }

    job = _job_manager.create(
        analysis_id=analysis_id,
        image_path=str(tmp_path),
        request_data=request_data,
        generate_grad_cam=generate_grad_cam,
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        status=job.status,
        created_at=datetime.fromtimestamp(job.created_at),
        modality=ModalityInfo(
            decision=None,
            reason="Analysis queued.",
            model=None,
            latency_ms=None,
        ),
        verification=VerificationInfo(
            available=False,
            model=None,
            latency_ms=None,
            error=None,
            agreement=None,
        ),
        labels=[],
        inference_latency_ms=None,
        total_latency_ms=None,
        report_available=False,
        grad_cam_available=False,
        contact_sheet_available=False,
        contact_sheet_url=None,
        disclaimer="AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical confidence.",
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, api_key: str | None = Security(get_api_key)) -> AnalysisResponse:
    """Get the status and results of an analysis."""
    if _inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend services not initialized.",
        )

    job = _inference_service.get_job(analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found.",
        )

    disclaimer = "AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical confidence."

    if job.status == JobStatus.QUEUED:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=job.status,
            created_at=datetime.fromtimestamp(job.created_at),
            modality=ModalityInfo(
                decision=None,
                reason="Queued for processing.",
                model=None,
                latency_ms=None,
            ),
            verification=VerificationInfo(
                available=False,
                model=None,
                latency_ms=None,
                error=None,
                agreement=None,
            ),
            labels=[],
            inference_latency_ms=None,
            total_latency_ms=None,
            report_available=False,
            grad_cam_available=False,
            contact_sheet_available=False,
            contact_sheet_url=None,
            disclaimer=disclaimer,
        )

    if job.status == JobStatus.PROCESSING:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=job.status,
            created_at=datetime.fromtimestamp(job.created_at),
            modality=ModalityInfo(
                decision=None,
                reason="Processing.",
                model=None,
                latency_ms=None,
            ),
            verification=VerificationInfo(
                available=False,
                model=None,
                latency_ms=None,
                error=None,
                agreement=None,
            ),
            labels=[],
            inference_latency_ms=None,
            total_latency_ms=None,
            report_available=False,
            grad_cam_available=False,
            contact_sheet_available=False,
            contact_sheet_url=None,
            disclaimer=disclaimer,
        )

    if job.status == JobStatus.FAILED:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=job.status,
            created_at=datetime.fromtimestamp(job.created_at),
            modality=ModalityInfo(
                decision="technical_failure",
                reason=job.error or "Analysis failed.",
                model=None,
                latency_ms=None,
            ),
            verification=VerificationInfo(
                available=False,
                model=None,
                latency_ms=None,
                error=job.error or "Analysis failed.",
                agreement=None,
            ),
            labels=[],
            inference_latency_ms=None,
            total_latency_ms=None,
            report_available=False,
            grad_cam_available=False,
            contact_sheet_available=False,
            contact_sheet_url=None,
            disclaimer=disclaimer,
        )

    # COMPLETED
    result = job.result or {}
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=job.status,
        created_at=datetime.fromtimestamp(job.created_at),
        modality=ModalityInfo(
            decision=result.get("modality", {}).get("decision", "unknown"),
            reason=result.get("modality", {}).get("reason"),
            model=result.get("modality", {}).get("model"),
            latency_ms=result.get("modality", {}).get("latency_ms"),
        ),
        labels=[
            LabelScore(
                label=l["label"],
                score=l.get("score"),
                severity=l["severity"],
                supported=l["supported"],
            )
            for l in result.get("labels", [])
        ],
        verification=VerificationInfo(
            available=result.get("verification", {}).get("available", False),
            model=result.get("verification", {}).get("model"),
            latency_ms=result.get("verification", {}).get("latency_ms"),
            error=result.get("verification", {}).get("error"),
            agreement=result.get("verification", {}).get("agreement"),
        ),
        inference_latency_ms=result.get("inference_latency_ms"),
        total_latency_ms=result.get("total_latency_ms"),
        report_available=result.get("report_available", False),
        grad_cam_available=result.get("grad_cam_available", False),
        contact_sheet_available=result.get("grad_cam_available", False),
        contact_sheet_url=(
            f"/api/v1/report/{analysis_id}/contact-sheet"
            if result.get("grad_cam_available") else None
        ),
        disclaimer=disclaimer,
        v7_report=result.get("v7_report"),
    )
