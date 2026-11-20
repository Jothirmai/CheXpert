"""Backend configuration for Chest Xpert FastAPI service.

All credentials are read from environment variables only. Nothing is hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEST_XPERT_ROOT = PROJECT_ROOT.parent  # /home/doli/Chest Xpert

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.environ.get("CHESTXPERT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("CHESTXPERT_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.environ.get("CHESTXPERT_WORKERS", "1")))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.environ.get("CHESTXPERT_CORS_ORIGINS", "*").split(",")
            if o.strip()
        ]
    )


SERVER_CONFIG = ServerConfig()

# ---------------------------------------------------------------------------
# Security / Auth
# ---------------------------------------------------------------------------
@dataclass
class SecurityConfig:
    api_key: str | None = field(default_factory=lambda: os.environ.get("CHESTXPERT_API_KEY"))
    require_auth: bool = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_REQUIRE_AUTH", "false").lower()
        == "true"
    )


SECURITY_CONFIG = SecurityConfig()

# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_name: str = "torchxrayvision-densenet121-res224-all"
    weights: str = "densenet121-res224-all"
    input_resolution: int = 224
    device: str = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_DEVICE", "cuda:0")
    )
    max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_MAX_CONCURRENT_JOBS", "1"))
    )
    job_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("CHESTXPERT_JOB_TIMEOUT_S", "300"))
    )
    grad_cam_enabled: bool = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_GRAD_CAM_ENABLED", "true").lower()
        == "true"
    )


MODEL_CONFIG = ModelConfig()

# ---------------------------------------------------------------------------
# Storage / artifacts
# ---------------------------------------------------------------------------
@dataclass
class StorageConfig:
    temp_dir: Path = Path(
        os.environ.get("CHESTXPERT_TEMP_DIR", str(PROJECT_ROOT / "temp"))
    )
    reports_dir: Path = Path(
        os.environ.get("CHESTXPERT_REPORTS_DIR", str(PROJECT_ROOT / "reports"))
    )
    artifact_ttl_s: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_ARTIFACT_TTL_S", "86400"))
    )
    max_upload_mb: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_MAX_UPLOAD_MB", "50"))
    )
    max_dimension: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_MAX_DIMENSION", "4096"))
    )


STORAGE_CONFIG = StorageConfig()

# ---------------------------------------------------------------------------
# LLM (optional, Stages B/C)
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    api_url: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LLM_API_URL")
    )
    api_token: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LLM_API_TOKEN")
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LLM_MODEL_ID", "")
    )
    enabled: bool = field(
        default_factory=lambda: bool(os.environ.get("CHESTXPERT_LLM_API_URL"))
    )


LLM_CONFIG = LLMConfig()

# ---------------------------------------------------------------------------
# Modality validation
# ---------------------------------------------------------------------------
@dataclass
class ModalityConfig:
    enabled: bool = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_MODALITY_ENABLED", "true").lower()
        == "true"
    )
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_HF_TOKEN")
    )


MODALITY_CONFIG = ModalityConfig()


def ensure_dirs() -> None:
    STORAGE_CONFIG.temp_dir.mkdir(parents=True, exist_ok=True)
    STORAGE_CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)


ensure_dirs()
