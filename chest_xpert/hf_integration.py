"""Hugging Face integration helpers for Chest Xpert.

Prepares the repository contents (README, model card, config, tests) for a
private Hugging Face repository. Does NOT upload third-party weights as the
user's own model, and does not hardcode or request access tokens.

The user authenticates separately via `huggingface-cli login` in the remote
terminal and decides whether/when to publish.
"""

from __future__ import annotations

from pathlib import Path

from chest_xpert.config import (
    APP_TITLE,
    DISCLAIMER,
    MODEL_CONFIG,
    PROJECT_ROOT,
)

HF_README = """\
# {title}

{subtitle}

## ⚠️ Important attribution

This repository contains **original software engineering work** (the Chest Xpert
inference application, UI, report generator, evaluation scripts, and model
registry). It does **not** contain a model trained by the repository author.

Chest Xpert loads **pre-trained weights created by other researchers** from
[torchxrayvision](https://github.com/mlmed/torchxrayvision). Those weights are
referenced (and downloaded at runtime) rather than redistributed here, unless the
upstream license permits redistribution and there is a genuine reason to host
them. Publishing this repository does **not** mean the underlying pre-trained
model was trained by me.

## Supported labels (18)

The selected pre-trained model — torchxrayvision `DenseNet-121`
(`densenet121-res224-all`) — emits all 18 requested chest X-ray findings:

Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema,
Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass,
Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum.

Scores are raw model outputs, **not calibrated probabilities or clinical
confidence**.

## Models and Methodology

See `docs/models_and_methodology.md` for per-model documentation, training
details (original authors' training, not mine), label mappings, licenses,
limitations, and cited sources.

## Setup

```bash
# Activate the existing environment (do not create a new one)
source /home/doli/anaconda3/bin/activate hct-lsr

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run chest_xpert/app.py
```

## Hugging Face publishing (private by default)

1. Log in from the remote terminal (your token is never stored in this repo):

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your HF token when prompted. It is saved under ~/.cache/huggingface/token.
   ```

2. Create a private repository:

   ```bash
   huggingface-cli repo create chest-xpert --private
   ```

3. Upload the repository files (README, model card, code, config, tests).
   Do **not** upload large third-party weight files unless the upstream license
   permits it.

4. Keep the repository **private** until you explicitly choose to publish it.

## Disclaimer

{discount}
""".format(
    title=APP_TITLE,
    subtitle=(
        "Chest X-ray multi-label findings inference using existing pre-trained "
        "models — no training, no fine-tuning."
    ),
    discount=DISCLAIMER,
)

MODEL_CARD = """\
---
language: en
license: mit
tags:
- chest-x-ray
- medical-imaging
- multi-label-classification
- grad-cam
- pretrained
- pytorch
---

# Chest Xpert — Model Card

## Intended use

Research and educational inference of 18 chest X-ray findings from a single
frontal chest radiograph. Not a medical device. Outputs require review by a
qualified radiologist.

## Model architecture

- Backbone: DenseNet-121 (pre-trained weights from torchxrayvision)
- Input: 224 × 224 single-channel grayscale, normalized to approx. [-1024, 1024]
- Outputs: 18 independent pathology scores (sigmoid)

## Upstream pre-trained checkpoint

- Name: torchxrayvision `densenet121-res224-all`
- Training datasets (documented by torchxrayvision): NIH ChestX-ray14,
  PadChest, CheXpert, MIMIC-CXR, ChestX-ray8, Google, OpenI, Kaggle/RSNA
- Weights URL: {weights_url}
- Paper: https://arxiv.org/abs/2002.02497
- Library: https://github.com/mlmed/torchxrayvision

The checkpoint is **not** a model trained by the repository author. It is
referenced at runtime and is not bundled in this repository.

## Labels

All 18 requested findings are supported by the upstream checkpoint. No outputs
are fabricated or remapped from unrelated tasks.

## Limitations

- Scores are not calibrated probabilities.
- Training data mixes datasets with different label definitions/uncertainty.
- Domain shift possible for unseen scanners/institutions.
- Not a substitute for clinical diagnosis.

{cross_ref}
""".format(
    weights_url=MODEL_CONFIG.weights_url,
    cross_ref=DISCLAIMER,
)

REQUIREMENTS = """\
torch>=2.2.0
torchvision>=0.17.0
torchxrayvision>=1.5.4
numpy
pillow
opencv-python
scikit-learn
scipy
pandas
streamlit
reportlab
pydicom
"""


def write_hf_files(target_dir: str | Path) -> dict:
    """Write README, model card and requirements into target_dir.

    Returns a dict of written file paths. Does not authenticate or push to HF.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    readme = target / "README.md"
    readme.write_text(HF_README)
    written["README.md"] = readme

    card = target / "MODEL_CARD.md"
    card.write_text(MODEL_CARD)
    written["MODEL_CARD.md"] = card

    req = target / "requirements.txt"
    req.write_text(REQUIREMENTS)
    written["requirements.txt"] = req

    return written


def auth_instructions() -> str:
    return (
        "To authenticate with Hugging Face from this remote terminal, run:\n"
        "  pip install huggingface_hub\n"
        "  huggingface-cli login\n"
        "Enter your HF access token when prompted. It is stored locally under "
        "~/.cache/huggingface/token and is never sent in chat or hardcoded in "
        "source code. If you are not authenticated, publishing will fail — that "
        "is expected; keep the repo private until you explicitly publish."
    )
