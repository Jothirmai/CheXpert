"""Sample-dataset smoke test for Chest Xpert.

Runs Stage 0 (file validation + modality) and Stage A (disease classifier)
against a set of test images:
  * 1 real chest X-ray (bundled sample_frontal.png)
  * 1 optional real chest X-ray downloaded from a public source
  * Synthetic fixtures: color photograph, grayscale document, cartoon-like,
    corrupted file, empty file

Results are written to tests/sample_dataset_results.json. No training runs.
No evaluation metrics are claimed — this is a functional smoke test only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from chest_xpert.config import MODEL_CONFIG
from chest_xpert.inference import InferenceEngine
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.modality import ModalityValidator
from chest_xpert.preprocessing import validate_image, preprocess

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT = Path(__file__).resolve().parent / "sample_dataset_results.json"
CXR_SAMPLE = SAMPLES / "sample_frontal.png"


def _synth(name: str, arr: np.ndarray) -> Path:
    p = SAMPLES / name
    Image.fromarray(arr).save(p)
    return p


def build_fixtures() -> dict[str, Path]:
    """Build license-free synthetic test fixtures."""
    fixtures: dict[str, Path] = {}

    # 1. Real chest X-ray (bundled sample).
    fixtures["valid_chest_xray"] = CXR_SAMPLE

    # 2. Color photograph (high saturation -> should be rejected by modality).
    color = np.zeros((224, 224, 3), dtype=np.uint8)
    color[:, :, 0] = 210  # red
    color[:, :, 1] = 60
    color[:, :, 2] = 60
    fixtures["color_photograph"] = _synth("fixture_color_photo.png", color)

    # 3. Grayscale document-like image (low saturation -> uncertain).
    doc = np.full((224, 224, 3), 170, dtype=np.uint8)
    d = ImageDraw.Draw(Image.fromarray(doc))
    # draw some dark text-like rectangles
    img = Image.fromarray(doc)
    draw = ImageDraw.Draw(img)
    for i in range(5):
        draw.rectangle([20, 20 + i * 30, 200, 35 + i * 30], fill=60)
    fixtures["grayscale_document"] = _synth("fixture_document.png", np.array(img))

    # 4. Cartoon-like (flat colors, high edges).
    cartoon = np.zeros((224, 224, 3), dtype=np.uint8)
    cartoon[40:184, 40:184] = [230, 230, 240]
    cartoon[60:160, 60:160] = [40, 40, 50]
    fixtures["cartoon"] = _synth("fixture_cartoon.png", cartoon)

    # 5. Corrupted file (not an image).
    bad = SAMPLES / "fixture_corrupted.bin"
    bad.write_bytes(b"\x89PNG\r\n\x1a\x00garbage-not-a-real-image-payload")
    fixtures["corrupted_file"] = bad

    # 6. Empty file.
    empty = SAMPLES / "fixture_empty.png"
    empty.write_bytes(b"")
    fixtures["empty_file"] = empty

    return fixtures


def run() -> None:
    engine = InferenceEngine(model_name=MODEL_CONFIG.name, device="cpu")
    validator = ModalityValidator()

    results: dict[str, dict] = {}

    for name, path in build_fixtures().items():
        entry: dict = {"path": str(path), "exists": path.exists()}
        t0 = time.perf_counter()

        # Stage 0: file validation.
        val = validate_image(path)
        entry["file_validation_ok"] = val.ok
        entry["file_validation_reason"] = val.reason
        entry["file_validation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if not val.ok:
            entry["stage"] = "rejected_at_file_validation"
            results[name] = entry
            continue

        # Stage 0: modality.
        t1 = time.perf_counter()
        mod = validator.validate(path)
        t2 = time.perf_counter()
        entry["modality_decision"] = mod.decision
        entry["modality_reason"] = mod.reason
        entry["modality_model"] = mod.model_id
        entry["modality_technical_failure"] = mod.is_technical_failure
        entry["modality_ms"] = round((t2 - t1) * 1000.0, 2)

        # Modality gate: only accepted (or explicitly uncertain + proceed) runs
        # disease classification. For this smoke test we only classify accepted.
        if mod.is_accepted:
            # Stage A: disease classifier.
            t3 = time.perf_counter()
            pp = preprocess(path, "cpu")
            t_prep = time.perf_counter()
            result = engine.predict(pp.tensor)
            t4 = time.perf_counter()
            mappings = map_outputs_to_labels(result.raw, result.model_name)
            entry["stage"] = "classified"
            entry["device"] = result.device
            entry["preprocess_ms"] = round((t_prep - t3) * 1000.0, 2)
            entry["inference_ms"] = round((t4 - t_prep) * 1000.0, 2)
            entry["total_ms"] = round((t4 - t3) * 1000.0, 2)
            entry["scores"] = {
                m.requested_label: (
                    round(m.score, 4) if (m.supported and m.score is not None) else None
                )
                for m in mappings
            }
            entry["supported_count"] = len(result.supported_labels)
            entry["unsupported_labels"] = result.unsupported_labels
        else:
            entry["stage"] = "rejected_at_modality"
            entry["classifier_skipped"] = True

        results[name] = entry

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")
    # Print a concise summary.
    for name, e in results.items():
        print(
            f"{name:24s} | stage={e.get('stage'):20s} | "
            f"modality={e.get('modality_decision','—'):12s} | "
            f"total_ms={e.get('total_ms','—')}"
        )


if __name__ == "__main__":
    run()
