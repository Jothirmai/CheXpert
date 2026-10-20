"""Image loading, validation and preprocessing for Chest Xpert.

Preprocessing follows the documented torchxrayvision recipe:
  * load PNG/JPG/DICOM
  * convert to single-channel grayscale
  * normalize pixels to roughly [-1024, 1024]
  * resize to the model's native resolution (224 x 224), bilinear, antialiased
  * produce a batch-ready tensor [1, 1, 224, 224]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import torchxrayvision as xrv
from torchxrayvision import utils as xrv_utils

from chest_xpert.config import (
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_MB,
    MODEL_CONFIG,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    size_mb: float = 0.0
    dims: Tuple[int, int] | None = None
    channels: int | None = None


def validate_image(path: str | Path) -> ValidationResult:
    """Validate file type, size and image dimensions before inference."""
    path = Path(path)
    if not path.exists():
        return ValidationResult(ok=False, reason="File does not exist.")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return ValidationResult(
            ok=False,
            reason=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}.",
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        return ValidationResult(
            ok=False,
            reason=f"File too large ({size_mb:.1f} MB > {MAX_IMAGE_MB} MB).",
        )

    try:
        img = load_image(path)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, reason=f"Failed to read image: {exc}")

    if img.ndim == 2:
        h, w = img.shape
        channels = 1
    elif img.ndim == 3:
        h, w = img.shape[0], img.shape[1]
        channels = img.shape[2]
    else:
        return ValidationResult(
            ok=False, reason=f"Unexpected image dimensionality: {img.ndim}."
        )

    return ValidationResult(
        ok=True,
        size_mb=round(size_mb, 3),
        dims=(int(h), int(w)),
        channels=int(channels),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a float32 numpy array.

    Supports PNG/JPEG via scikit-image and DICOM via pydicom/torchxrayvision.
    Returns a 2D single-channel array scaled to [-1024, 1024].
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".dcm", ".dicom"}:
        # torchxrayvision's DICOM reader (requires pydicom, now installed).
        arr = xrv_utils.read_xray_dcm(str(path), voi_lut=False, fix_monochrome=True)
        # read_xray_dcm returns [1, H, W]
        return np.asarray(arr[0], dtype=np.float32)

    import skimage.io

    img = skimage.io.imread(str(path))
    if img.ndim == 3:
        # Convert RGB/RGBA to single channel by taking the first channel and
        # normalizing consistently with torchxrayvision (max 255 for 8-bit).
        img = img[:, :, 0]
    arr = xrv_utils.normalize(img, 255, reshape=False)
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing for the model
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedImage:
    tensor: torch.Tensor  # [1, 1, H, W], on the requested device
    original_size: Tuple[int, int]
    resized: bool


def preprocess(
    path: str | Path,
    device: str,
    resolution: int | None = None,
) -> PreprocessedImage:
    """Run documented torchxrayvision preprocessing and return a batch tensor."""
    if resolution is None:
        resolution = MODEL_CONFIG.input_resolution

    path = Path(path)
    arr = load_image(path)  # [H, W] in [-1024, 1024]
    original_size = (int(arr.shape[0]), int(arr.shape[1]))

    # Ensure 2D single channel.
    if arr.ndim != 2:
        arr = arr[:, :, 0]

    tensor = torch.from_numpy(arr).float()  # [H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    resized = False
    if tensor.shape[2] != resolution or tensor.shape[3] != resolution:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(resolution, resolution), mode="bilinear", antialias=True
        )
        resized = True

    tensor = tensor.to(device)
    return PreprocessedImage(tensor=tensor, original_size=original_size, resized=resized)
