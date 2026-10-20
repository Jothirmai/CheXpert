"""Chest Xpert FastAPI backend entrypoint.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Or from the project root:
    source /home/doli/anaconda3/bin/activate hct-lsr
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1 import admin, analyses, health, report
from backend.config import SERVER_CONFIG, STORAGE_CONFIG, ensure_dirs
from backend.security.auth import get_api_key
from backend.services.cleanup_service import cleanup_service
from backend.services.inference_service import InferenceService, inference_service
from backend.services.job_manager import JobManager

# Singletons created at startup.
job_manager = JobManager(inference_service)

# Wire cross-module references.
analyses.init_routers(inference_service, job_manager)
admin.init_admin_routers(job_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup.
    ensure_dirs()
    # Load the classifier model eagerly so /ready reflects reality.
    ok, err = await inference_service.ensure_model_loaded()
    if not ok:
        # Don't crash — the first request will retry. Log the error.
        print(f"[startup] Model preload deferred: {err}")
    yield
    # Shutdown: release GPU resources.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


app = FastAPI(
    title="Chest Xpert API",
    description=(
        "Chest X-ray multi-label findings inference using pre-trained models. "
        "No training, no fine-tuning. AI-generated research draft — not a clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrict to actual frontend origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER_CONFIG.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*", "X-API-Key"],
)

# Routes.
app.include_router(health.router, prefix="/api/v1")
app.include_router(analyses.router, prefix="/api/v1/analyses")
app.include_router(report.router, prefix="/api/v1/report")
app.include_router(admin.router, prefix="/api/v1/admin")


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Chest Xpert API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "ready": "/api/v1/ready",
    }
