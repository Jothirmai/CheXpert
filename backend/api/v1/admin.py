"""Administrative endpoints: diagnostics, cleanup, cache sizes.

These are protected by API key when authentication is enabled.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Security, status

from backend.security.auth import get_api_key
from backend.services.cleanup_service import cleanup_service
from backend.services.inference_service import inference_service
from backend.services.job_manager import JobManager
from backend.services.resource_manager import MONITOR
from backend.schemas import DiagnosticsResponse

router = APIRouter(tags=["admin"])


# Module-level job manager reference (wired in main.py).
_job_manager: JobManager | None = None


def init_admin_routers(job_manager: JobManager) -> None:
    global _job_manager
    _job_manager = job_manager


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(api_key: str | None = Security(get_api_key)) -> DiagnosticsResponse:
    """Return safe aggregate operational metrics (no case data)."""
    snap = MONITOR.snapshot("diagnostics")
    active = _job_manager.active_count() if _job_manager else 0
    queued = _job_manager.queued_count() if _job_manager else 0

    load_info: dict = {}
    if inference_service._load_snapshot:
        load_info = inference_service._load_snapshot

    return DiagnosticsResponse(
        gpu_name=snap.gpu_name,
        gpu_total_vram_mb=snap.gpu_total_mb,
        gpu_free_vram_mb_before_load=(
            load_info.get("before", {}).gpu_free_mb
            if isinstance(load_info.get("before"), object)
            and hasattr(load_info.get("before"), "gpu_free_mb")
            else None
        ),
        gpu_free_vram_mb_after_load=(
            load_info.get("after", {}).gpu_free_mb
            if isinstance(load_info.get("after"), object)
            and hasattr(load_info.get("after"), "gpu_free_mb")
            else None
        ),
        pytorch_allocated_mb=snap.torch_allocated_mb,
        pytorch_reserved_mb=snap.torch_reserved_mb,
        peak_allocated_mb=snap.peak_allocated_mb,
        host_ram_used_gb=snap.host_ram_used_gb,
        host_ram_available_gb=snap.host_ram_available_gb,
        model_load_time_ms=load_info.get("load_time_ms"),
        active_jobs=active,
        queued_jobs=queued,
        disk_usage=MONITOR.disk_usage(),
        gpu_processes=MONITOR.gpu_process_info(),
    )


@router.post("/cleanup")
def cleanup(
    dry_run: bool = True,
    ttl_s: int | None = None,
    api_key: str | None = Security(get_api_key),
):
    """Run or simulate cleanup of expired temp artifacts (dry-run by default)."""
    return cleanup_service.run(ttl_s=ttl_s, dry_run=dry_run)


@router.get("/cache-sizes")
def cache_sizes(api_key: str | None = Security(get_api_key)):
    """Return read-only cache/directory sizes."""
    return cleanup_service.cache_sizes()
