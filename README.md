# Chest Xpert

A research prototype that accepts chest radiographs, computes 14 CNN finding scores, and drafts an editable report with a vision-language model. The React interface supports image validation, report review, templates, history, review acknowledgement, and PDF export. Generated text requires qualified human review; this is not a clinically validated diagnostic system.

## What is included

- `backend/`: FastAPI service and review workflow.
- `frontend-react/`: React application.
- `chest_xpert/`: inference, preprocessing, explainability, and report logic.
- `cliq_rrg/` and `scripts/`: training and evaluation code.
- `tests/` and `samples/`: code tests and small test fixtures.
- `checkpoints/chexpert_plus_densenet201_v6/best_model.pth` and `checkpoints/chexpert_plus_impression_320_v12/best_model.pth`: selected CNN ensemble weights.
- `checkpoints/report_lora_chexpert_plus_v1/adapter_config.json` and `adapter_model.safetensors`: selected VLM LoRA adapter.
- `checkpoints/active_models.json`: selected model paths, hashes, evaluation context, and limitations.

The dataset, intermediate training checkpoints, reports, and videos are excluded. The public [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) base model and [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) modality model are downloaded into the Hugging Face cache during setup. They are not stored in this Git repository.

## Requirements

Use Linux, Python 3.10, Node.js 18 or newer, and an NVIDIA CUDA GPU. The application has been exercised on a 32 GB V100 with 4-bit Qwen loading. Full inference requires a working CUDA driver and substantial free disk space for the roughly 16 GB Qwen base model and CLIP weights. On Ubuntu, PDF export may also require `libcairo2`, `libpango-1.0-0`, `libpangocairo-1.0-0`, and `libgdk-pixbuf-2.0-0`.

The CUDA 11.8 wheel command below follows [PyTorch's 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/). Use a different supported PyTorch CUDA build if your driver requires it, while keeping the tested Python package versions compatible.

## Set up from a fresh clone

```bash
git clone https://github.com/Jothirmai/CheXpert.git
cd CheXpert
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-app.txt
python scripts/prepare_app.py --download-models
cd frontend-react
npm ci
cd ..
```

`prepare_app.py` verifies the bundled CNN and LoRA hashes, downloads the two public base models at the revisions selected by the app, and checks that their files are complete. The first download is large and may take time. The models are public; a Hugging Face token is normally unnecessary. If your environment needs authentication, log in with `hf auth login` locally. Do not commit tokens or `.env` files.

### Start the backend

From the repository root with `.venv` active:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The backend warms CLIP, both CNN checkpoints, and Qwen with the selected LoRA adapter at startup. Wait for [the readiness endpoint](http://127.0.0.1:8000/api/v1/ready) to return `"ready": true` before uploading an image. The first startup can take several minutes.

### Start the frontend

In another terminal:

```bash
cd CheXpert
npm --prefix frontend-react run dev
```

Open `http://localhost:5173`. For a browser on another machine, set `VITE_API_URL` to the backend origin before starting Vite and configure `CHESTXPERT_CORS_ORIGINS` on the backend. The default local workflow does not require an API key. To enable the demo key or provision individual reviewers, set environment variables described in `backend/.env.example`; the file is an example and is not loaded automatically.

### Verify the app

```bash
curl http://127.0.0.1:8000/api/v1/ready
python scripts/verify_application.py --output /tmp/chest-xpert-smoke
```

The smoke script checks invalid-image rejection and a real sample-image report through the API. It writes its report artifacts outside the repository. It is an engineering smoke test, not clinical validation.

## Training and evaluation

The training entry points are `scripts/train_chexpert_densenet201_v6.py`, `scripts/train_chexpert_plus_impression.py`, and `scripts/train_report_adapter.py`; evaluation and audit scripts are also in `scripts/`. Running training or reproducing published scores requires separately acquired datasets, patient-disjoint manifests, substantial compute, and the original pretrained base models. No training dataset is shipped here. The selected CNN's five-finding CheXpert score in `active_models.json` is an exploratory measured result on an inspected cohort, not an independent final test or a state-of-the-art claim.

## Review limitations

CNN scores are uncalibrated model outputs. The VLM report is a draft and may omit or misstate findings. Electronic acknowledgement records entered reviewer details but does not independently verify clinical credentials. Review each image and report before use.
