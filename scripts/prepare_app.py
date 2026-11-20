"""Check bundled weights and fetch the public base models needed by the app.

Run from the repository root after installing requirements-app.txt:
    python scripts/prepare_app.py --download-models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chest_xpert.model_revisions import (  # noqa: E402
    CLIP_CONFIG_REVISION,
    CLIP_MODEL_ID,
    CLIP_SAFETENSORS_REVISION,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_local_artifacts() -> None:
    selection = json.loads((ROOT / "checkpoints/active_models.json").read_text())
    cnn = selection["cnn"]
    for relpath in cnn["paths"]:
        name = Path(relpath).parent.name
        path = ROOT / relpath
        if not path.is_file():
            raise SystemExit(f"Missing selected CNN checkpoint: {relpath}")
        actual = sha256(path)
        if actual != cnn["sha256"][name]:
            raise SystemExit(f"CNN checkpoint hash mismatch: {relpath}")
        print(f"OK: {relpath} ({path.stat().st_size:,} bytes)")

    adapter_path = ROOT / selection["report_adapter"]["path"]
    config = adapter_path / "adapter_config.json"
    weights = adapter_path / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise SystemExit(f"Missing selected VLM adapter files in {adapter_path}")
    base = json.loads(config.read_text())["base_model_name_or_path"]
    if base != QWEN_MODEL_ID:
        raise SystemExit(f"Adapter expects {base}, but the app selects {QWEN_MODEL_ID}")
    if sha256(weights) != selection["report_adapter"]["adapter_sha256"]:
        raise SystemExit("Selected VLM adapter hash mismatch")
    print(f"OK: {weights.relative_to(ROOT)} ({weights.stat().st_size:,} bytes)")


def check_or_download_models(download: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    mode = "download" if download else "local cache check"
    print(f"Qwen base model ({mode}): {QWEN_MODEL_ID}@{QWEN_REVISION}")
    qwen = Path(snapshot_download(
        repo_id=QWEN_MODEL_ID,
        revision=QWEN_REVISION,
        local_files_only=not download,
        allow_patterns=[
            "config.json", "generation_config.json", "chat_template.json",
            "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
            "merges.txt", "vocab.json", "model.safetensors.index.json",
            "model-*.safetensors",
        ],
    ))
    index = json.loads((qwen / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (qwen / name).is_file()]
    if missing:
        raise SystemExit(f"Qwen download is incomplete: {missing}")
    print(f"OK: all {len(set(index['weight_map'].values()))} Qwen weight shards present")

    print(f"CLIP configuration ({mode}): {CLIP_MODEL_ID}@{CLIP_CONFIG_REVISION}")
    clip = Path(snapshot_download(
        repo_id=CLIP_MODEL_ID,
        revision=CLIP_CONFIG_REVISION,
        local_files_only=not download,
        allow_patterns=[
            "config.json", "preprocessor_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "merges.txt", "vocab.json",
        ],
    ))
    if not (clip / "config.json").is_file() or not (clip / "preprocessor_config.json").is_file():
        raise SystemExit("CLIP configuration download is incomplete")
    safe = Path(hf_hub_download(
        repo_id=CLIP_MODEL_ID,
        filename="model.safetensors",
        revision=CLIP_SAFETENSORS_REVISION,
        local_files_only=not download,
    ))
    print(f"OK: CLIP safetensors weights ({safe.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true", help="Download the public Qwen and CLIP base models")
    args = parser.parse_args()
    check_local_artifacts()
    check_or_download_models(args.download_models)
    print("Application model files are ready. Start the backend and check /api/v1/ready.")


if __name__ == "__main__":
    main()
