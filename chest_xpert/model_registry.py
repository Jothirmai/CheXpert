"""Model registry for Chest Xpert.

The registry describes each available pre-trained model, the labels it
genuinely supports, and the mapping from the model's raw outputs to the 18
requested chest X-ray labels.

Design principles:
  * Never fabricate predictions. Labels a model does not emit are marked
    ``unsupported`` and are never mapped from unrelated outputs.
  * Separate documented facts (from the original library/model card) from
    project-specific annotations.
  * A single model (torchxrayvision DenseNet-121 "all" weights) supports all 18
    requested labels, so no model combination is strictly required. Additional
    models can be registered later if desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from chest_xpert.config import LABELS_18, ModelConfig, MODEL_CONFIG

# torchxrayvision uses underscores in some pathology names.
TORCHXRAYVISION_LABELS = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]


@dataclass
class LabelSupport:
    """Support status for one of the 18 requested labels."""

    requested_label: str
    supported: bool
    model_output_name: Optional[str] = None  # name the model actually emits
    source: str = ""  # where this mapping was documented
    notes: str = ""


@dataclass
class ModelInfo:
    """Documented facts about one pre-trained model."""

    name: str
    architecture: str
    parameter_count: Optional[str]
    authors: str
    organization: str
    publication: str
    repository: str
    training_datasets: str
    training_objective: str
    loss_function: str
    input_dimensions: str
    preprocessing: str
    multi_label_head: str
    supported_labels: List[LabelSupport]
    license: str
    permitted_uses: str
    limitations: str
    evaluation_datasets: str
    published_metrics: str
    weights_url: str
    trained_by_this_project: bool = False


def _txv_label_support() -> List[LabelSupport]:
    """Build label support for the torchxrayvision DenseNet-121 'all' weights.

    The model emits exactly the 18 pathologies in TORCHXRAYVISION_LABELS, which
    correspond one-to-one to the 18 requested labels (torchxrayvision writes
    "Pleural_Thickening" where the request uses "Pleural Thickening").
    """
    support: List[LabelSupport] = []
    for requested in LABELS_18:
        # Canonical mapping: normalize spaces/underscores for comparison.
        normalized_requested = requested.replace(" ", "_")
        model_name = normalized_requested if normalized_requested in TORCHXRAYVISION_LABELS else requested
        supported = model_name in TORCHXRAYVISION_LABELS
        support.append(
            LabelSupport(
                requested_label=requested,
                supported=supported,
                model_output_name=model_name if supported else None,
                source=(
                    "torchxrayvision model_urls['densenet121-res224-all']['labels'] "
                    "(github.com/mlmed/torchxrayvision)"
                ),
                notes=(
                    "Emitted directly by the model; not remapped from another output."
                    if supported
                    else "Not emitted by this model."
                ),
            )
        )
    return support


def registry() -> Dict[str, ModelInfo]:
    """Return the model registry keyed by model name.

    Facts about the original pre-trained model are taken from torchxrayvision's
    model.py / model card and the cited paper. Details unknown from the source
    are explicitly marked as unknown rather than invented.
    """
    txv_support = _txv_label_support()
    all_supported = all(ls.supported for ls in txv_support)

    infos: Dict[str, ModelInfo] = {}
    infos[MODEL_CONFIG.name] = ModelInfo(
        name=MODEL_CONFIG.name,
        architecture="DenseNet-121 (Densely Connected Convolutional Networks)",
        parameter_count="~8 million (DenseNet-121; exact count not documented "
        "by torchxrayvision)",
        authors="torchxrayvision maintainers (mlmed); weights described alongside "
        "the CheXpert baseline work",
        organization="mlmed / Stanford ML Group (CheXpert) — see cited sources",
        publication=(
            "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels "
            "and Comparison of Standard and Deep Learning Baselines, Irvin et al., "
            "https://arxiv.org/abs/2002.02497 ; torchxrayvision library "
            "(github.com/mlmed/torchxrayvision)"
        ),
        repository="https://github.com/mlmed/torchxrayvision",
        training_datasets=(
            "Combined: NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, "
            "ChestX-ray8, Google, OpenI, Kaggle / RSNA (per torchxrayvision "
            "weights description: nih-pc-chex-mimic_ch-google-openi-kaggle-rsna)"
        ),
        training_objective=(
            "Multi-label binary classification of chest radiograph findings"
        ),
        loss_function="Binary cross-entropy (per torchxrayvision training recipe; "
        "exact formulation not documented in the library)",
        input_dimensions="224 x 224, single-channel grayscale",
        preprocessing=(
            "Resize to 224x224 (bilinear), single channel, normalize pixels to "
            "approximately [-1024, 1024] via xrv.utils.normalize"
        ),
        multi_label_head=(
            "Single linear classifier producing one logit per pathology, followed "
            "by sigmoid and optional operating-point normalization (op_threshs). "
            "Outputs are independent per label (not calibrated probabilities)."
        ),
        supported_labels=txv_support,
        license=MODEL_CONFIG.license,
        permitted_uses=(
            "Research / educational inference using pre-trained weights. Do not "
            "redistribute the weights as your own trained model."
        ),
        limitations=(
            "Scores are not calibrated probabilities. Training data mixes multiple "
            "datasets with different label definitions and uncertainty handling; "
            "domain shift possible for images from unseen institutions/scanners. "
            "Not a substitute for radiologist review."
        ),
        evaluation_datasets=(
            "Evaluated on held-out splits of the constituent training datasets by "
            "the original authors (AUROC reported); not re-evaluated by this "
            "project unless the evaluation module is run."
        ),
        published_metrics=(
            "Per-pathology operating-point thresholds (op_threshs) and PPV@80 "
            "thresholds ship with the weights; published per-label AUROC values "
            "are not documented in the library and are not reproduced here."
        ),
        weights_url=MODEL_CONFIG.weights_url,
        trained_by_this_project=False,
    )
    return infos


def supported_labels_for(model_name: str) -> List[str]:
    """Display names of labels supported by the given model."""
    infos = registry()
    info = infos[model_name]
    return [ls.requested_label for ls in info.supported_labels if ls.supported]


def unsupported_labels_for(model_name: str) -> List[str]:
    """Display names of labels not supported by the given model."""
    infos = registry()
    info = infos[model_name]
    return [ls.requested_label for ls in info.supported_labels if not ls.supported]


def label_support_summary(model_name: str) -> Dict[str, bool]:
    """Map each of the 18 requested labels to a supported/not-supported bool."""
    infos = registry()
    info = infos[model_name]
    return {ls.requested_label: ls.supported for ls in info.supported_labels}
