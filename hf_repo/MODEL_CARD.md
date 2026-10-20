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
- Weights URL: https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
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

AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical confidence.
