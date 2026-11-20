# Chest Xpert

Research prototype for chest radiograph classification and editable report drafting. The active application is a FastAPI backend and React frontend. A chest X-ray upload passes image validation, CNN inference, attribution generation, and VLM report drafting. A reviewer can edit the draft, apply a report template, record a review acknowledgement, and export a PDF.

## Repository contents

- `backend/`: API, review workflow, authentication, and report export.
- `chest_xpert/`: image preprocessing, CNN/VLM inference, model selection, explainability, and evaluation.
- `cliq_rrg/`: experimental report-generation and training modules.
- `scripts/`: data preparation, CNN/VLM training, evaluation, and audits.
- `frontend-react/`: active web interface.
- `tests/`: source-level and workflow tests, with small synthetic fixtures in `samples/`.
- `checkpoints/active_models.json`: selected-model metadata and expected local checkpoint paths.

Datasets, downloaded model weights, trained checkpoints, reports, and videos are intentionally excluded. A fresh clone needs separately obtained datasets and model artifacts before inference or training can run. The selected CNN uses two locally trained DenseNet-201 checkpoints; the VLM uses a locally trained LoRA adapter on Qwen2.5-VL-7B-Instruct. Training scripts document their own pretrained initialization and data requirements. Do not interpret model outputs as a clinical diagnosis.

## Run the app

Use Python 3.10 with a CUDA-capable PyTorch installation. Install the Python dependencies from `requirements.txt` and `backend/requirements.txt`, plus the Transformers, PEFT, and BitsAndBytes versions compatible with the local model environment. Place the required model artifacts at the paths in `checkpoints/active_models.json`, or set `CHESTXPERT_CNN_CHECKPOINT` for a compatible single-checkpoint configuration. Copy `backend/.env.example` to a local `.env` and configure access as needed; never commit credentials.

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend-react
npm ci
npm run dev
```

The frontend runs at `http://localhost:5173`. The backend readiness endpoint is `/api/v1/ready`.

## Train and evaluate

The main training and evaluation entry points are in `scripts/`. For usage and required arguments:

```bash
python scripts/train_cnn_corrected.py --help
python scripts/train_report_adapter.py --help
python scripts/evaluate_selected_ensemble_all14.py --help
python scripts/evaluate_selected_vlm_reserved_test.py --help
```

Training and evaluation require the relevant dataset access, patient-disjoint manifests, local weights, and a suitable GPU. The repository excludes those data and artifacts. `checkpoints/active_models.json` records experimental selection and its limitations; the referenced measurements do not establish clinical validity or a state-of-the-art claim.

## Review and safety

Generated reports are drafts. CNN scores are uncalibrated outputs. Electronic acknowledgement records entered reviewer information but does not verify professional identity. Clinical use requires qualified human review and independent validation.
