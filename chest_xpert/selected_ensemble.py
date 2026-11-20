"""Frozen CheXpert DenseNet201 ensemble used as report-generation evidence.

The five-finding AUROC belongs to CheXpert studies. IU X-Ray scores are
out-of-domain supporting signals and have no measured IU AUROC claim.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / 'checkpoints/chexpert_plus_densenet201_v6/best_model.pth'
V12 = ROOT / 'checkpoints/chexpert_plus_impression_320_v12/best_model.pth'


class SelectedEnsembleEngine:
    """0.5*v6(image) + 0.25*v12(image) + 0.25*v12(flip(image))."""

    def __init__(self, device='cuda:0'):
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Selected CNN ensemble requires CUDA')
        self.device = device
        from chest_xpert.model_selection import selection
        selected = selection().get('cnn', {})
        self.labels = selected.get('labels', [])
        if len(self.labels) != 14 or len(set(self.labels)) != 14:
            raise ValueError('Selected CNN requires 14 unique labels in checkpoints/active_models.json')
        self.model_name = 'CheXpert DenseNet201 v6 + v12 flip TTA ensemble'
        self.checkpoint_paths = [V6, V12]
        self.checkpoint_sha256 = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.checkpoint_paths}
        if selected.get('architecture') == 'densenet201_v6_v12_tta' and selected.get('sha256') != self.checkpoint_sha256:
            raise RuntimeError('Selected CNN checkpoint hashes changed')
        self.fingerprint = hashlib.sha256(json.dumps(self.checkpoint_sha256, sort_keys=True).encode()).hexdigest()
        self.v6 = self._load(V6)
        self.v12 = self._load(V12)
        # Attribution is from v12; scores are from the exact ensemble above.
        self.explanation_model = self.v12
        self.model = self.v12
        self.resolution = 320
        self.pixel_scale = 1.0
        self.crop = 'resize'
        self.transform224 = self._transform(224)
        self.transform320 = self._transform(320)

    def _load(self, path):
        model = models.densenet201(weights=None, num_classes=14)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        model.to(self.device).eval().requires_grad_(False)
        return model

    @staticmethod
    def _transform(size):
        return transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def prepare(self, path):
        path = Path(path)
        if path.suffix.lower() in {'.dcm', '.dicom'}:
            from chest_xpert.preprocessing import load_image
            gray = np.uint8(np.clip((load_image(path) + 1024) * (255 / 2048), 0, 255))
            rgb = Image.fromarray(gray, mode='L').convert('RGB')
        else:
            with Image.open(path) as source:
                rgb = source.convert('RGB')
        return self.transform224(rgb).unsqueeze(0).to(self.device), self.transform320(rgb).unsqueeze(0).to(self.device)

    def predict_pair(self, pair):
        start = time.perf_counter()
        x224, x320 = pair
        with torch.inference_mode():
            scores = (self.v6(x224).sigmoid() * .5
                      + self.v12(x320).sigmoid() * .25
                      + self.v12(x320.flip(-1)).sigmoid() * .25)[0]
        raw = dict(zip(self.labels, map(float, scores.cpu().tolist())))
        return SimpleNamespace(raw=raw, model_name=self.model_name,
                               model_outputs=self.labels, unsupported_labels=[],
                               inference_only_ms=(time.perf_counter()-start)*1000)

    def predict_paths(self, paths):
        pairs = [self.prepare(path) for path in paths]
        predictions = [self.predict_pair(pair) for pair in pairs]
        raw = {name: float(np.mean([p.raw[name] for p in predictions])) for name in self.labels}
        return raw, pairs

    def contact_sheet(self, pairs, scores, output):
        from dataclasses import replace
        from chest_xpert.explainability import GradCAM, generate_contact_sheet, save_contact_sheet
        top = sorted(self.labels, key=lambda name: scores[name], reverse=True)[:6]
        cam = GradCAM(self.explanation_model, target_layer_name='features.norm5')
        results = []
        displays = []
        try:
            for index, (_, x320) in enumerate(pairs):
                # Use the preprocessed 320-pixel view for the overlay geometry.
                normalized = x320[0].detach().cpu().numpy()
                rgb = np.clip(normalized * np.array([.229, .224, .225])[:, None, None]
                              + np.array([.485, .456, .406])[:, None, None], 0, 1)
                display = rgb.mean(axis=0)
                displays.append(display)
                for label in top:
                    item = cam.generate(x320, label, self.labels, display)
                    if len(pairs) > 1:
                        item = replace(item, target_label=f'View {index+1}: {label}')
                    results.append(item)
        finally:
            cam.close()
        sheet = generate_contact_sheet(results, displays[0],
                                       cols=6 if len(pairs)>1 else 3,
                                       rows=len(pairs) if len(pairs)>1 else 2)
        save_contact_sheet(sheet, str(output))
        return str(output)
