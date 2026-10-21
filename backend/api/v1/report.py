"""Report export endpoint."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Security, status
from fastapi.responses import FileResponse

from backend.security.auth import get_api_key
from backend.services.inference_service import inference_service
from backend.schemas import ReportExportRequest, ReportResponse

router = APIRouter(tags=["report"])


@router.get("/{analysis_id}/report", response_model=ReportResponse)
def get_report(
    analysis_id: str,
    format: str = "pdf",
    api_key: str | None = Security(get_api_key),
) -> ReportResponse:
    """Export the report for a completed analysis as PDF or HTML."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend services not initialized.",
        )

    job = inference_service.get_job(analysis_id)
    if job is None or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found or not completed.",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis {analysis_id} is {job.status}; report not ready.",
        )

    try:
        fmt, file_path = inference_service.generate_report_export(
            analysis_id=analysis_id,
            fmt=format,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {exc}",
        )

    return ReportResponse(
        analysis_id=analysis_id,
        format=fmt,
        download_url=f"/api/v1/report/{analysis_id}/download?format={fmt}",
        generated_at=datetime.utcnow(),
        disclaimer="AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.",
    )


@router.get("/{analysis_id}/download")
def download_report(
    analysis_id: str,
    format: str = "pdf",
    api_key: str | None = Security(get_api_key),
):
    """Download the generated report file."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend services not initialized.",
        )

    from backend.config import STORAGE_CONFIG

    file_path = STORAGE_CONFIG.reports_dir / f"{analysis_id}.{format}"
    if not file_path.exists():
        # Try generating on demand.
        try:
            _, file_path = inference_service.generate_report_export(
                analysis_id=analysis_id, fmt=format
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found for {analysis_id}.",
            )

    media_type = "application/pdf" if format == "pdf" else "text/html"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"chest_xpert_report_{analysis_id}.{format}",
    )


@router.get("/{analysis_id}/contact-sheet")
def download_contact_sheet(
    analysis_id: str,
    api_key: str | None = Security(get_api_key),
):
    """Download the Grad-CAM contact sheet image."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend services not initialized.",
        )

    job = inference_service.get_job(analysis_id)
    if job is None or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found or not completed.",
        )

    from backend.config import STORAGE_CONFIG

    contact_sheet_path = STORAGE_CONFIG.temp_dir / f"{analysis_id}_gradcam_contact_sheet.png"
    if not contact_sheet_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact sheet not found for {analysis_id}.",
        )

    return FileResponse(
        path=str(contact_sheet_path),
        media_type="image/png",
        filename=f"chest_xpert_contact_sheet_{analysis_id}.png",
    )
