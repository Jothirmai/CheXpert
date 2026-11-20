"""Pydantic request/response schemas for the Chest Xpert API.

Only user-facing, safe fields are exposed. No internal prompts, raw model
reasoning, or filesystem paths are returned to the client.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ModalityDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    TECHNICAL_FAILURE = "technical_failure"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "chest-xpert"
    version: str = "1.0.0"
    model_loaded: bool = False
    device: str = "cpu"


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, str]


# ---------------------------------------------------------------------------
# Analysis request / response
# ---------------------------------------------------------------------------
class AnalysisCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: Optional[str] = Field(
        None, max_length=64, description="Optional patient identifier (not stored permanently)"
    )
    exam_date: Optional[str] = Field(None, max_length=32)
    modality: Optional[str] = Field("Chest X-ray", max_length=32)
    view: Optional[str] = Field(None, max_length=16)
    institution: Optional[str] = Field(None, max_length=64)
    referring_physician: Optional[str] = Field(None, max_length=64)
    generate_grad_cam: bool = Field(
        False, description="Whether to generate a Grad-CAM contact sheet"
    )


class LabelScore(BaseModel):
    score: Optional[float] = None
    label: str
    severity: str = "not assessed"
    supported: bool


class ModalityInfo(BaseModel):
    decision: Optional[ModalityDecision] = None
    reason: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None


class VerificationInfo(BaseModel):
    available: bool
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    agreement: Optional[str] = None  # e.g. "3 consistent · 1 inconsistent · 2 uncertain"


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: JobStatus
    created_at: datetime
    modality: ModalityInfo
    verification: VerificationInfo
    labels: list[LabelScore]
    inference_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    report_available: bool = False
    grad_cam_available: bool = False
    contact_sheet_available: bool = False
    contact_sheet_url: Optional[str] = None
    disclaimer: str
    v7_report: Optional[str] = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
class ReportExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = Field("pdf", pattern="^(pdf|html)$")
    impression: Optional[str] = Field(None, max_length=2000)
    reviewer_name: Optional[str] = Field(None, max_length=64)
    review_status: Optional[str] = Field(None, max_length=32)
    notes: Optional[str] = Field(None, max_length=1000)


class ReportResponse(BaseModel):
    analysis_id: str
    format: str
    download_url: str
    generated_at: datetime
    disclaimer: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: str


# ---------------------------------------------------------------------------
# Diagnostics (admin only, safe aggregates)
# ---------------------------------------------------------------------------
class DiagnosticsResponse(BaseModel):
    gpu_name: Optional[str] = None
    gpu_total_vram_mb: Optional[float] = None
    gpu_free_vram_mb_before_load: Optional[float] = None
    gpu_free_vram_mb_after_load: Optional[float] = None
    pytorch_allocated_mb: Optional[float] = None
    pytorch_reserved_mb: Optional[float] = None
    peak_allocated_mb: Optional[float] = None
    host_ram_used_gb: Optional[float] = None
    host_ram_available_gb: Optional[float] = None
    model_load_time_ms: Optional[float] = None
    classifier_inference_time_ms: Optional[float] = None
    grad_cam_time_ms: Optional[float] = None
    report_generation_time_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    active_jobs: int = 0
    queued_jobs: int = 0
    disk_usage: Optional[dict] = None
    gpu_processes: Optional[list[dict]] = None
