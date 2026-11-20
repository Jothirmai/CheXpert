"""Configuration for the Chest Xpert application.

All paths are resolved relative to this file so the package works regardless of
the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from chest_xpert.model_selection import selected_adapter

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = PROJECT_ROOT / "samples"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Default sample chest X-ray (inspected from the local server only; not part of
# the HCT-LSR application code). Used for smoke tests and demos.
DEFAULT_SAMPLE_IMAGE = SAMPLES_DIR / "sample_frontal.png"

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
@dataclass
class DeviceConfig:
    """Device selection for inference.

    Prefers CUDA when available with sufficient free memory; otherwise falls
    back to CPU. The shared Tesla V100 on this server is often near capacity,
    so CPU is the reliable default.
    """
    preferred: str = "auto"   # "auto" | "cuda" | "cpu"

    def resolve(self) -> str:
        import torch
        if self.preferred == "cpu":
            return "cpu"
        if self.preferred == "cuda" or self.preferred == "auto":
            if torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info()
                    free_gib = free / (1024 ** 3)
                    if free_gib >= 2.0:
                        return "cuda"
                except Exception:
                    pass
        return "cpu"


DEVICE_CONFIG = DeviceConfig()

# ---------------------------------------------------------------------------
# The 18 requested chest X-ray labels (canonical display names)
# ---------------------------------------------------------------------------
LABELS_18 = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# Canonical label -> display label (torchxrayvision uses underscores).
LABEL_DISPLAY_MAP = {
    "Pleural_Thickening": "Pleural Thickening",
    "Lung Opacity": "Lung Opacity",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
}

# ---------------------------------------------------------------------------
# Selected pretrained model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = "torchxrayvision-densenet121-res224-all"
    weights: str = "densenet121-res224-all"
    architecture: str = "DenseNet-121"
    input_resolution: int = 224
    in_channels: int = 1
    # Pixel scaling expected by torchxrayvision: images normalized to [-1024, 1024]
    pixel_min: float = -1024.0
    pixel_max: float = 1024.0
    cache_dir: str = str(Path.home() / ".torchxrayvision" / "models_data")
    # Source of the pre-trained weights (not created by this project).
    weights_url: str = (
        "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
        "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-"
        "rot45-tr15-sc15-seed0-best.pt"
    )
    # Paper describing the weights/training combination.
    paper_url: str = "https://arxiv.org/abs/2002.02497"
    library_repo: str = "https://github.com/mlmed/torchxrayvision"
    license: str = "MIT (torchxrayvision library)"
    # This project did not train the model; it only loads pre-trained weights.
    trained_by_this_project: bool = False

MODEL_CONFIG = ModelConfig()


# ---------------------------------------------------------------------------
# Stage 0 — Modality validation configuration
# ---------------------------------------------------------------------------
@dataclass
class ModalityConfig:
    """Configuration for chest X-ray modality validation.

    Uses CLIP zero-shot classification against a small set of candidate
    modalities. The decision rule is a documented heuristic, NOT a validated
    clinical threshold. Accuracy is not claimed.
    """
    enabled: bool = True
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = os.environ.get("CHESTXPERT_DEVICE", "cuda:0")
    # Candidate modalities for zero-shot comparison.
    candidate_labels: list[str] = field(default_factory=lambda: [
        "a chest X-ray",
        "a photograph",
        "a CT scan",
        "an MRI scan",
        "a document",
        "a screenshot",
        "a cartoon or drawing",
        "another kind of medical image",
    ])
    # Decision policy (heuristic, undocumented-as-validated):
    accepted_min_prob: float = 0.45
    accepted_margin_over_best_other: float = 0.05
    rejected_max_prob: float = 0.30


MODALITY_CONFIG = ModalityConfig()

# ---------------------------------------------------------------------------
# Stage B / C — LLM configuration
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Configuration for optional remote LLM inference (Stages B and C).

    Credentials are read exclusively from environment variables; they are never
    hardcoded in source, logs, or the UI. If no endpoint/token is configured,
    LLM stages are marked unavailable (not fabricated).
    """
    api_url: str | None = field(default_factory=lambda: os.environ.get("CHESTXPERT_LLM_API_URL"))
    api_token: str | None = field(default_factory=lambda: os.environ.get("CHESTXPERT_LLM_API_TOKEN"))
    model_id: str = field(default_factory=lambda: os.environ.get("CHESTXPERT_LLM_MODEL_ID", ""))
    timeout_s: float = 30.0
    enabled: bool = field(default_factory=lambda: bool(os.environ.get("CHESTXPERT_LLM_API_URL")))


LLM_CONFIG = LLMConfig()


# ---------------------------------------------------------------------------
# Stage B — Qwen VLM configuration (free Hugging Face options)
# ---------------------------------------------------------------------------
@dataclass
class LLMModelConfig:
    """Report-model configuration.

    The active report workflow requires transformers_local, a CUDA vision-language
    model, the original X-ray, a labeled Grad-CAM contact sheet and CNN scores.
    Text-only models and remote text-generation calls cannot replace missing vision
    evidence. Credentials are read from environment variables.
    """
    backend: str = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LLM_BACKEND", "transformers_local")
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get(
            "CHESTXPERT_LLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
    )
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_HF_TOKEN")
    )
    # vLLM server URL (only used when backend="vllm_server")
    vllm_url: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_VLLM_URL", "")
    )
    # Force CPU loading (only for small text models; VL models need GPU)
    force_cpu: bool = field(
        default_factory=lambda: os.environ.get(
            "CHESTXPERT_LLM_FORCE_CPU", "false"
        ).lower()
        == "true"
    )
    # Use 4-bit quantization for large VL models (fits on 32 GB V100)
    load_in_4bit: bool = field(
        default_factory=lambda: os.environ.get(
            "CHESTXPERT_LLM_LOAD_IN_4BIT", "true" if selected_adapter() else "false"
        ).lower()
        == "true"
    )


LLM_MODEL_CONFIG = LLMModelConfig()


# ---------------------------------------------------------------------------
# Stage B — LoRA fine-tuning configuration (optional)
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration for fine-tuning the Qwen VLM.

    LoRA adds small trainable adapter matrices to the frozen base model, so
    fine-tuning is fast and memory-efficient while the base weights stay
    unchanged. At inference time the adapters are merged (or applied on the
    fly) so output behaviour is preserved for inputs outside the fine-tuning
    distribution, while task-specific responses are accelerated/improved.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LORA_ENABLED", "true" if selected_adapter() else "false").lower() == "true"
    )
    adapter_path: str | None = field(
        default_factory=lambda: os.environ.get("CHESTXPERT_LORA_ADAPTER_PATH", selected_adapter())
    )
    r: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_LORA_R", "16"))
    )
    lora_alpha: int = field(
        default_factory=lambda: int(os.environ.get("CHESTXPERT_LORA_ALPHA", "32"))
    )
    lora_dropout: float = field(
        default_factory=lambda: float(os.environ.get("CHESTXPERT_LORA_DROPOUT", "0.05"))
    )
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


LORA_CONFIG = LoRAConfig()


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------
APP_TITLE = "Chest Xpert"
APP_SUBTITLE = "Chest X-ray multi-label findings — pre-trained model inference"
DISCLAIMER = (
    "AI-generated research draft requiring review by a qualified radiologist. "
    "Not a final clinical diagnosis. Prediction scores are not calibrated "
    "probabilities or clinical confidence."
)
MAX_IMAGE_MB = 50
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
