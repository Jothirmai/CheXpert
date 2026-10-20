# Models and Methodology — Chest Xpert

This page documents the pre-trained model used by Chest Xpert, its training
(methodology of the **original authors**, not this project), label mappings,
licenses, limitations, and cited sources.

It deliberately separates **documented facts** (from the original library,
model card, paper, and repository) from **details unknown from those sources**,
which are marked as unknown rather than invented.

## 1. Selected pre-trained model

| Item | Value |
|------|-------|
| Model name | torchxrayvision `DenseNet-121` (`densenet121-res224-all`) |
| Architecture | DenseNet-121 ("Densely Connected Convolutional Networks") |
| Parameter count | ~8 million (DenseNet-121; exact count not documented by torchxrayvision) |
| Input dimensions | 224 × 224, single-channel grayscale |
| Pre-trained weights | Referenced at runtime from `~/.torchxrayvision/models_data/` (already cached on this server) |
| Weights URL | https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt |
| Library | torchxrayvision 1.5.4 — https://github.com/mlmed/torchxrayvision |
| License | MIT (torchxrayvision library) |
| Trained by this project? | **No.** This project only loads pre-trained weights created by other researchers. |

## 2. Original authors and publication

The torchxrayvision library is maintained by **mlmed**. The `densenet121-res224-all`
weights are described alongside the CheXpert baseline work:

- **Paper:** Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with
  Uncertainty Labels and Comparison of Standard and Deep Learning Baselines,"
  https://arxiv.org/abs/2002.02497
- **Architecture source:** Huang et al., "Densely Connected Convolutional
  Networks," https://arxiv.org/abs/1608.06993 (DenseNet-121)
- **Repository:** https://github.com/mlmed/torchxrayvision

The weights description states they were trained on a combination of public
chest X-ray datasets: NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR,
ChestX-ray8, Google, OpenI, and Kaggle/RSNA. (Note: torchxrayvision's own
description lists "rsna" while the weights filename references "kaggle"; this
discrepancy is in the upstream library and is not resolved here.)

## 3. Original training methodology (documented facts)

The following is what the original torchxrayvision documentation / paper
supports. Details not documented are marked **unknown**.

### Dataset preparation
- Training data: a combination of NIH ChestX-ray14, PadChest, CheXpert,
  MIMIC-CXR, ChestX-ray8, Google, OpenI, and Kaggle/RSNA chest radiographs.
- Labels: 18 chest radiograph pathologies (see Section 5).
- **Unknown:** exact train/validation/test split ratios, patient-level split
  procedure, handling of uncertain labels per dataset, image source
  deduplication, and any cohort filtering.

### Preprocessing (documented by torchxrayvision)
- Input images are resized (bilinear interpolation) to the native resolution
  the model was trained on: **224 × 224**.
- Single-channel (grayscale).
- Pixels normalized to approximately **[-1024, 1024]** via `xrv.utils.normalize`
  (for 8-bit images: `(2 * (img / 255) - 1) * 1024`).
- **Unknown:** any additional augmentation (rotation, flip, etc.) used during
  original training. The weights filename contains `rot45` and `tr15`, hinting
  at rotation/translation augmentation, but this is not explicitly documented.

### Loss and optimization
- Training objective: **multi-label binary classification** of chest findings
  (one independent output per pathology).
- Loss function: **binary cross-entropy** is the standard for this recipe, but
  the exact loss formulation is **not documented** in torchxrayvision.
- **Unknown:** optimizer, learning rate, batch size, number of epochs, learning
  rate schedule, weight decay, early stopping, checkpoint selection criterion.
  These hyperparameters are not documented in the library and are **not**
  invented here.

### Multi-label prediction head
- A single linear classifier produces **one logit per pathology**.
- Outputs pass through a **sigmoid**, then optional **operating-point
  normalization** (`op_threshs`) that rescales each score so its operating
  threshold maps to 0.5.
- Outputs are **independent per label** (not a softmax distribution) and are
  **not calibrated probabilities**.

## 4. Labels supported (18 requested)

The `densenet121-res224-all` weights emit all 18 requested findings, one-to-one:

1. Atelectasis
2. Consolidation
3. Infiltration
4. Pneumothorax
5. Edema
6. Emphysema
7. Fibrosis
8. Effusion
9. Pneumonia
10. Pleural Thickening (emitted as `Pleural_Thickening`)
11. Cardiomegaly
12. Nodule
13. Mass
14. Hernia
15. Lung Lesion
16. Fracture
17. Lung Opacity
18. Enlarged Cardiomediastinum

**Unsupported labels: none** — all 18 requested labels are supported by the
selected model. No outputs are fabricated or remapped from unrelated tasks.

## 5. Original evaluation and published metrics

- **Evaluation datasets:** held-out splits of the constituent training datasets
  (per the original authors). The exact held-out test sets and patient-level
  separation are **not documented** in torchxrayvision.
- **Published metrics:** the weights ship with per-pathology operating-point
  thresholds (`op_threshs`) and PPV@80 thresholds (`ppv80_thres`). Published
  per-label AUROC/AUPRC values are **not documented** in the library and are
  **not reproduced** by this project unless the evaluation module is run on a
  real test split.
- **Known concerns:** training data mixes datasets with different label
  definitions, prevalence, and uncertainty handling (e.g., CheXpert's uncertain
  labels vs. NIH's binary labels); domain shift is possible for images from
  unseen institutions or scanners.

## 6. Limitations

- Scores are raw model outputs, **not calibrated probabilities or clinical
  confidence**.
- Training data mixes multiple public datasets with different label definitions
  and uncertainty conventions.
- Domain shift may affect performance on scanners/institutions not represented
  in training.
- Not a medical device; not a substitute for review by a qualified radiologist.
- Grad-CAM heatmaps are model explanations, **not proof** that a disease is
  present.

## 7. Attribution boundary

| What | Who did it |
|------|------------|
| Original model training | torchxrayvision / CheXpert researchers (original authors) |
| Chest Xpert application code | This project (original software engineering) |
| Inference / evaluation experiments | This project (when run) |
| Future fine-tuning | Not performed |

This project does **not** claim to have trained the pre-trained model, and does
**not** redistribute third-party weights as its own.

## 8. Sources

- torchxrayvision library & model card: https://github.com/mlmed/torchxrayvision
- DenseNet paper: https://arxiv.org/abs/1608.06993
- CheXpert paper: https://arxiv.org/abs/2002.02497
- Weights: https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
