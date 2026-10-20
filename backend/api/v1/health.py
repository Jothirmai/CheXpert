"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.config import MODEL_CONFIG
from backend.services.inference_service import inference_service
from backend.services.resource_manager import MONITOR
from backend.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok, _ = (
        inference_service.is_model_loaded(),
        None,
    )
    return HealthResponse(
        status="ok",
        model_loaded=inference_service.is_model_loaded(),
        device=(MODEL_CONFIG.device if MODEL_CONFIG.device != "auto" else "auto"),
    )


@router.get("/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    checks: dict[str, str] = {}
    all_ok = True

    # Model loadable?
    if inference_service.is_model_loaded():
        checks["model"] = "loaded"
    else:
        checks["model"] = "not loaded (will load on first request)"
        all_ok = False

    # Temp dir writable?
    try:
        from backend.config import STORAGE_CONFIG

        STORAGE_CONFIG.temp_dir.mkdir(parents=True, exist_ok=True)
        test_file = STORAGE_CONFIG.temp_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        checks["temp_dir"] = "writable"
    except Exception as exc:  # noqa: BLE001
        checks["temp_dir"] = f"not writable: {exc}"
        all_ok = False

    return ReadinessResponse(ready=all_ok, checks=checks)
