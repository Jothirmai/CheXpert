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
    checks: dict[str, str] = {"report_inputs": "required: cnn_scores, original_xray, gradcam_contact_sheet"}
    all_ok = True

    # Model loadable?
    if inference_service.is_model_loaded():
        checks["model"] = "loaded"
    else:
        checks["model"] = "not loaded (will load on first request)"
        all_ok = False

    from chest_xpert.modality import modality_validator
    if modality_validator._clip_model is not None:
        checks["clip"] = "loaded"
    else:
        checks["clip"] = modality_validator._clip_error or "not loaded"
        all_ok = False
    vision = inference_service._verification_engine
    if vision is not None and vision.client.backend == "transformers_local" and vision.client.is_configured():
        checks["vision"] = vision.client.model_id
    else:
        checks["vision"] = "not available"
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
