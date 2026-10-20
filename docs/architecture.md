# Chest Xpert — Architecture

## Modular architecture

```mermaid
flowchart TD
    subgraph UI [Streamlit UI]
        A[Dashboard] --> B[Upload & Inference]
        B --> C[Explainability Grad-CAM]
        B --> D[Report Generator]
        C --> D
        E[Models & Methodology]
        F[Model Comparison & Evaluation]
    end

    subgraph PKG [chest_xpert package]
        G[model_registry.py] - model registry + label support
        H[preprocessing.py] - load/validate/preprocess
        I[inference.py] - cached model + predict + latency
        J[label_mapping.py] - model outputs -> 18 labels
        K[explainability.py] - Grad-CAM
        L[report.py] - HTML/PDF report
        M[evaluation.py] - AUROC/AUPRC
        N[hf_integration.py] - HF repo prep
    end

    B --> I
    I --> H
    I --> J
    C --> K
    D --> L
    M --> I
    G --> J
```

## Data flow (inference)

1. User uploads a chest X-ray (PNG/JPG/DICOM).
2. `preprocessing.validate_image` checks extension, size, dimensions.
3. `preprocessing.preload` loads + normalizes to [-1024, 1024], resizes to 224×224.
4. `inference.InferenceEngine` loads `torchxrayvision` DenseNet-121 once (cached),
   runs `torch.no_grad()` forward, measures latency.
5. `label_mapping.map_outputs_to_labels` maps raw outputs to the 18 requested
   labels; unsupported labels are listed explicitly (none for this model).
6. `explainability.GradCAM` produces a heatmap for a user-selected supported
   finding (target layer: `features.norm5`).
7. `report.generate_html` / `generate_pdf` produce an editable report, stamped
   as an AI-generated research draft requiring radiologist review.

## Device selection

- Prefers CUDA when a GPU with sufficient free memory is available.
- Falls back to CPU otherwise. On this server the Tesla V100 is shared and
  often near capacity, so CPU is the reliable default.
- Measured latency on CPU for one 224×224 image: ~471 ms inference, ~473 ms
  total (sample image). Sub-500 ms was investigated but is not promised.

## Tests

`tests/test_core.py` covers: label mapping, all-18 support, unsupported labels,
preprocessing shape/range, validation of bad extensions, report HTML disclaimer,
report PDF generation, and failure handling.

Run with: `pytest tests/`

## Troubleshooting

- **CUDA out of memory:** the V100 is shared with other processes. The app falls
  back to CPU automatically; inference still completes (~0.5 s/image).
- **Model load failure:** ensure `~/.torchxrayvision/models_data/` is writable
  and internet is available for the first weight download (weights are already
  cached on this server).
- **Grad-CAM error:** requires grad-capable model; the engine uses `no_grad`
  (not `inference_mode`) so the same instance supports Grad-CAM.
- **PDF/report errors:** `reportlab` must be installed (see requirements.txt).
