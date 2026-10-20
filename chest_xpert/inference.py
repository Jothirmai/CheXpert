"""Inference engine for Chest Xpert.

Loads the selected pre-trained model once and reuses it across requests.
Runs under ``torch.inference_mode()`` with explicit device selection and
end-to-end latency measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torchxrayvision as xrv

from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG
from chest_xpert.model_registry import registry


@dataclass
class InferenceResult:
    scores: Dict[str, float]          # requested label -> score (supported only)
    raw: Dict[str, float]             # model output name -> score
    model_outputs: List[str]          # model pathology names in order
    device: str
    latency_ms: float                 # preprocessing + inference total
    inference_only_ms: float          # model forward pass only
    model_name: str
    supported_labels: List[str]
    unsupported_labels: List[str]


class InferenceEngine:
    """Singleton-style engine that loads the model once."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or MODEL_CONFIG.name
        self.device = device or DEVICE_CONFIG.resolve()
        self._model = None
        self._input_resolution = MODEL_CONFIG.input_resolution
        self._registry_info = registry()[self.model_name]

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        import torch
        import torchxrayvision as xrv

        torch.cuda.empty_cache()
        model = xrv.models.DenseNet(weights=MODEL_CONFIG.weights)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, tensor: torch.Tensor) -> InferenceResult:
        """Run inference on a preprocessed [1, 1, H, W] tensor.

        Automatically falls back from CUDA to CPU if CUDA runs out of memory
        (common on shared GPUs).
        """
        import torch

        device = self.device
        tensor = tensor.to(device)

        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(tensor)
            t1 = time.perf_counter()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device != "cpu":
                # Fall back to CPU.
                torch.cuda.empty_cache()
                self.device = "cpu"
                self._model = None
                engine = self._load_model()
                self._model = engine
                tensor = tensor.to("cpu")
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = self._model(tensor)
                t1 = time.perf_counter()
                device = "cpu"
            else:
                raise

        inference_only_ms = (t1 - t0) * 1000.0

        scores = logits.detach().cpu().numpy()[0]
        model_outputs = list(self._model.pathologies)

        raw = {name: float(score) for name, score in zip(model_outputs, scores)}

        # Map to requested labels; never fabricate unsupported labels.
        scores_map: Dict[str, float] = {}
        supported: List[str] = []
        unsupported: List[str] = []
        info = self._registry_info
        for ls in info.supported_labels:
            if ls.supported and ls.model_output_name in raw:
                scores_map[ls.requested_label] = raw[ls.model_output_name]
                supported.append(ls.requested_label)
            else:
                unsupported.append(ls.requested_label)

        total_ms = inference_only_ms
        return InferenceResult(
            scores=scores_map,
            raw=raw,
            model_outputs=model_outputs,
            device=device,
            latency_ms=total_ms,
            inference_only_ms=inference_only_ms,
            model_name=self.model_name,
            supported_labels=supported,
            unsupported_labels=unsupported,
        )

    def reload(self, device: str | None = None) -> None:
        """Re-load the model (e.g. after switching device)."""
        if device is not None:
            self.device = device
        self._model = None
        _ = self.model
