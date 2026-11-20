"""Inference adapter for the project's trained 14-finding checkpoint."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from cliq_rrg.stage1 import Stage1CNN

ROOT = Path(__file__).resolve().parents[1]


class CheckpointEngine:
    def __init__(self, device="cuda:0", checkpoint=None, **kwargs):
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Check the NVIDIA driver and PyTorch CUDA installation.")
        torch.cuda.set_device(torch.device(device))
        self.device = device
        from chest_xpert.model_selection import selected_cnn
        path = Path(checkpoint) if checkpoint else selected_cnn()
        config = json.loads((path.parent / "training_config.json").read_text())
        self.labels = config["diseases"]
        self.resolution = config["resolution"]
        self.pixel_scale = float(config.get("pixel_scale", 1.0))
        self.crop = config.get("crop", "resize")
        self.config = config
        if not path.is_file():
            raise FileNotFoundError(f"Trained CNN checkpoint missing: {path}")
        import hashlib
        self.checkpoint_path = path.resolve()
        self.fingerprint = hashlib.sha256(path.read_bytes() + json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.model = Stage1CNN(num_diseases=len(self.labels), pretrained=False)
        self.model.alignment.disease_specific_queries = bool(config.get("disease_specific_queries", False))
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.model_name = config.get("model_name", "Chest Xpert trained Stage1CNN")

    def predict(self, tensor):
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)["disease_probs"][0]
        torch.cuda.synchronize(self.device)
        raw = dict(zip(self.labels, output.detach().cpu().tolist()))
        return SimpleNamespace(raw=raw, model_name=self.model_name,
            model_outputs=self.labels, unsupported_labels=[],
            inference_only_ms=(time.perf_counter()-start)*1000)


def prepare_checkpoint_image(path, resolution=224, pixel_scale=1.0, crop="resize"):
    """Shared checkpoint-specific preprocessing for training and inference."""
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import center_crop
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        from chest_xpert.preprocessing import load_image
        image = torch.from_numpy(load_image(path)).float().unsqueeze(0) / 1024.0
        if crop == "center":
            side = min(image.shape[-2:]); image = center_crop(image, [side, side])
        return torch.nn.functional.interpolate(image.unsqueeze(0),size=(resolution,resolution),mode="bilinear",antialias=True)[0] * pixel_scale
    with Image.open(path) as source:
        image = source.convert("RGB")
        if crop == "center":
            side = min(image.size); image = center_crop(image, [side, side])
        transform = transforms.Compose([transforms.Resize((resolution,resolution)),transforms.Grayscale(1),transforms.ToTensor(),transforms.Normalize([.5],[.5])])
        return transform(image) * pixel_scale
