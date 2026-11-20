"""Eager warm-up of the CLIP modality validator at backend startup.

Loads the CLIP zero-shot model once so the first analysis request doesn't
pay the ~6-7s model-loading penalty. Runs in a thread to avoid blocking
the async event loop.
"""

from __future__ import annotations

import asyncio

from chest_xpert.modality import modality_validator


async def warmup_modality() -> None:
    """Load the CLIP modality validator eagerly on the configured CUDA device."""

    def _load() -> None:
        validator = modality_validator
        # Force CLIP to load (Qwen only loads if HF_TOKEN is set).
        validator._ensure_clip_loaded()

    await asyncio.to_thread(_load)
