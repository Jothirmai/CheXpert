"""Explainability via Grad-CAM for Chest Xpert.

Grad-CAM highlights image regions the model attends to for a selected finding.
Target layer: the final feature block output of the DenseNet-121
(``model.features`` up to ``norm5``), the standard choice for Grad-CAM on
DenseNets.

The heatmap is a model explanation, not proof that a disease is present.
Grad-CAM is only generated for labels the model genuinely supports.

Uses Pillow + numpy (no OpenCV) to avoid extra heavy dependencies.

Implementation note: torchxrayvision's ``features2`` applies an in-place ReLU to
the target-layer output, which conflicts with autograd backward hooks. We
therefore avoid hooks entirely: activations come from ``model.features`` and
gradients are obtained with ``torch.autograd.grad`` w.r.t. those activations,
re-implementing the head tail (ReLU + global average pool + classifier) with a
non-in-place ReLU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from chest_xpert.config import MODEL_CONFIG


@dataclass
class GradCAMResult:
    heatmap: np.ndarray          # [H, W] float in [0, 1]
    overlay: np.ndarray          # [H, W, 3] uint8 heatmap over original image
    target_label: str
    target_layer: str
    class_index: int


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Approximate MATLAB JET colormap for values in [0, 1] -> [H, W, 3] in [0, 1]."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - abs(4.0 * v - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "features.norm5"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name

    def close(self) -> None:
        """No hooks to remove (hook-free implementation). Kept for API symmetry."""
        return None

    def _activations_and_gradients(
        self,
        input_tensor: torch.Tensor,
        class_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (activations, gradients) w.r.t. the target layer.

        activations: [1, C, Hf, Wf] output of model.features (norm5).
        gradients:   [1, C, Hf, Wf] d(target_score)/d(activations).
        """
        # Activations from the target layer (final feature block output).
        # model.features ends at norm5; no in-place ReLU downstream here.
        feats = self.model.features(input_tensor)  # [1, C, Hf, Wf]
        if not feats.requires_grad:
            feats = feats.requires_grad_(True)

        # Re-implement features2's head tail with a NON-in-place ReLU to avoid
        # the in-place ReLU / autograd conflict in torchxrayvision.
        out = F.relu(feats)                                   # non-in-place
        pooled = F.adaptive_avg_pool2d(out, (1, 1)).view(feats.size(0), -1)
        logits = self.model.classifier(pooled)                # [1, num_pathologies]
        target_score = logits[0, class_index]

        # Gradient of the target score w.r.t. the target-layer activations.
        (grads,) = torch.autograd.grad(
            target_score, feats, retain_graph=False
        )
        return feats.detach(), grads.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_label: str,
        model_outputs: List[str],
        original_image: np.ndarray,
    ) -> GradCAMResult:
        """Produce a Grad-CAM heatmap for ``target_label``."""
        if target_label not in model_outputs:
            raise ValueError(
                f"Label '{target_label}' is not a model output. "
                f"Available: {model_outputs}"
            )

        class_index = model_outputs.index(target_label)

        # Fresh clone in normal grad mode (no inference_mode) so autograd works.
        input_tensor = input_tensor.detach().clone()
        torch.set_grad_enabled(True)

        feats, grads = self._activations_and_gradients(input_tensor, class_index)

        # Global average pooling of gradients -> weights per channel.
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * feats).sum(dim=1, keepdim=True)  # [1, 1, Hf, Wf]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam_max = float(cam.max())
        if cam_max > 0:
            cam = cam / cam_max

        # Resize heatmap to original image size using Pillow.
        h, w = original_image.shape[:2]
        heatmap_img = Image.fromarray(np.uint8(255 * np.clip(cam, 0, 1)), mode="L")
        heatmap_resized = (
            np.array(heatmap_img.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
        )
        heatmap_resized = np.clip(heatmap_resized, 0, 1)

        overlay = self._overlay_heatmap(original_image, heatmap_resized)

        return GradCAMResult(
            heatmap=heatmap_resized,
            overlay=overlay,
            target_label=target_label,
            target_layer=self.target_layer_name,
            class_index=class_index,
        )

    @staticmethod
    def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay a [0,1] heatmap on a grayscale image; return uint8 RGB."""
        img = image.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        img_rgb = np.stack([img, img, img], axis=-1)

        heatmap_color = _jet_colormap(heatmap)
        overlay = (1.0 - alpha) * img_rgb + alpha * heatmap_color
        overlay = np.clip(overlay, 0, 1)
        return np.uint8(255 * overlay)
