# Chest Xpert — Complete Documentation

**Project:** Chest Xpert  
**Location:** `/home/doli/Chest Xpert`  
**Environment:** `/home/doli/anaconda3/envs/hct-lsr` (Python 3.10.21, reused — no new env created)  
**Status:** Built and tested — no training, no fine-tuning, no fabricated results.

---

## 1. Overview

Chest Xpert is a research inference application that predicts 18 chest X-ray findings
using **existing pre-trained models only**. It does not train or fine-tune any model.
It implements a 5-stage pipeline:

- **Stage 0:** Image & modality validation (file validation + chest X-ray vs. other modality check)
- **Stage A:** Pretrained disease classification (DenseNet-121)
- **Stage B:** LLM #1 prediction verification (optional remote VLM)
- **Stage C:** LLM #2 report generation (deterministic template fallback)
- **Stage D:** Human review & export (Streamlit UI)

Every report is stamped: **AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.**

---

## 2. Project structure

```
/home/doli/Chest Xpert/
├── chest_xpert/
│   ├── __init__.py
│   ├── config.py            # env, device, 18 labels, model, modality, LLM config
│   ├── preprocessing.py     # Stage 0 file validation + torchxrayvision preprocessing
│   ├── modality.py          # Stage 0 CLIP zero-shot + heuristic fallback
│   ├── inference.py         # Stage A cached DenseNet-121, no_grad, latency
│   ├── label_mapping.py     # model outputs -> 18 labels (no fabrication)
│   ├── verification.py      # Stage B LLM #1 structured verification
│   ├── explainability.py    # Grad-CAM (hook-free, features.norm5)
│   ├── report.py            # Stage C/D HTML + PDF report
│   ├── hf_integration.py    # HF repo prep (private by default)
│   ├── evaluation.py        # AUROC/AUPRC script (not run)
│   └── app.py               # Stage D Streamlit pipeline UI
├── tests/
│   ├── test_core.py                 # 9 core tests
│   ├── test_pipeline.py             # 12 pipeline tests
│   ├── test_sample_dataset.py       # sample-dataset smoke test
│   └── sample_dataset_results.json  # smoke-test results
├── docs/
│   ├── models_and_methodology.md
│   └── architecture.md
├── hf_repo/                       # README.md, MODEL_CARD.md, requirements.txt (unpublished)
├── samples/
│   ├── sample_frontal.png         # bundled real chest X-ray (224x224)
│   ├── fixture_color_photo.png    # synthetic color photo
│   ├── fixture_document.png       # synthetic grayscale document
│   ├── fixture_cartoon.png        # synthetic cartoon
│   ├── fixture_corrupted.bin      # synthetic corrupted file
│   └── fixture_empty.png          # synthetic empty file
├── requirements.txt
└── README.md
```

---

## 3. Supported labels (all 18)

The selected pre-trained model (torchxrayvision `densenet121-res224-all`) emits **all 18**
requested findings one-to-one. No labels are unsupported; none are fabricated or remapped.

| # | Label |
|---|-------|
| 1 | Atelectasis |
| 2 | Consolidation |
| 3 | Infiltration |
| 4 | Pneumothorax |
| 5 | Edema |
| 6 | Emphysema |
| 7 | Fibrosis |
| 8 | Effusion |
| 9 | Pneumonia |
| 10 | Pleural Thickening |
| 11 | Cardiomegaly |
| 12 | Nodule |
| 13 | Mass |
| 14 | Hernia |
| 15 | Lung Lesion |
| 16 | Fracture |
| 17 | Lung Opacity |
| 18 | Enlarged Cardiomediastinum |

Scores are raw model outputs (sigmoid), **not calibrated probabilities or clinical confidence**.

---

## 4. Model attribution

| Component | Source | Trained by this project? | License |
|-----------|--------|--------------------------|---------|
| Disease classifier | torchxrayvision DenseNet-121 (`densenet121-res224-all`) | **No** — weights by torchxrayvision / CheXpert researchers (Irvin et al., arXiv:2002.02497) | MIT (library) |
| Modality validator | Qwen2.5-VL (`Qwen/Qwen2.5-VL-7B-Instruct`, OpenAI/Qwen) + CLIP zero-shot + heuristic fallback | **No** | — |
| LLM verification/report | Optional remote service via `CHESTXPERT_LLM_API_URL` | **No** — not configured by default | — |

- Weights are **referenced**, not redistributed or claimed as your own.
- Hugging Face repository (in `hf_repo/`) is prepared **private by default**; you decide whether to publish.
- This project is original software engineering work; the pretrained components belong to their original authors.

**Environment upgrade:** `torch` was upgraded from 2.5.1+cu118 to 2.7.1+cu118 (and `torchvision` to 0.22.1+cu118) to enable CLIP modality detection. Verification confirmed:
- `torchxrayvision 1.5.4` still loads and runs the DenseNet-121 classifier correctly
- CLIP (`openai/clip-vit-base-patch32`) loads successfully on CPU
- All 21 tests pass after the upgrade

---

## 5. Stage-by-stage behavior

### Stage 0 — Image & modality validation

**File validation:** checks extension (`.png/.jpg/.jpeg/.dcm/.dicom`), readability, dimensions, and max size (50 MB). Corrupted, empty, unsupported, and malformed files are rejected safely.

**Modality validation:** determines whether the image appears to be a chest X-ray.
- **Primary:** Qwen2.5-VL (`Qwen/Qwen2.5-VL-7B-Instruct`) — a vision-language model that can actually inspect the image. Requires `CHESTXPERT_HF_TOKEN` environment variable (free Hugging Face read token). Runs on GPU if available, falls back to CPU.
- **Secondary:** CLIP zero-shot (`openai/clip-vit-base-patch32`) — text-image similarity fallback. Runs on CPU. Works but weaker on black-and-white images.
- **Tertiary:** grayscale-saturation heuristic — safety net if both Qwen and CLIP fail. Can reject color photos but **cannot confirm** an X-ray.
- Decision policy (heuristic, **not a validated clinical threshold**):
  - `accepted`: sufficient evidence the image is a chest X-ray
  - `rejected`: clear evidence it is not
  - `uncertain`: insufficient evidence
  - `technical_failure`: model unavailable / timeout / auth failure (distinguished from rejection)

**UI behavior on rejection:**
- Non-X-ray: *"Input rejected: The uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image to continue."*
- Ambiguous: *"Unable to verify the image type. Please upload a clear chest X-ray image and try again."*
- On either: classifier, LLM #1, LLM #2 are **not** run; no scores/report shown; user can retry; prior results are invalidated on new upload.
- Technical failure is reported separately and never claimed as "image invalid."

### Stage A — Pretrained disease classifier

- Loads DenseNet-121 once, reuses across requests, runs under `torch.no_grad()`.
- Input: 224×224 single-channel grayscale, normalized to ~[-1024, 1024].
- Returns scores only for supported labels; unsupported marked unavailable (none for this model).
- Records model identity, version, preprocessing, inference latency.
- Grad-CAM available for supported findings (target layer: `features.norm5`); heatmap = explanation, not proof of diagnosis.

### Stage B — LLM #1 prediction verification

- Optional remote vision-language LLM. **Unavailable unless** `CHESTXPERT_LLM_API_URL` (and token) are set in environment — never hardcoded.
- Two responsibilities: (A) image-quality/interpretability assessment, (B) per-pathology `consistent`/`inconsistent`/`uncertain` verification with rationale.
- Safeguards: never treats high score as proof; never claims Grad-CAM proves disease; disagreements preserved; malformed responses handled; marked as secondary assessment, not ground truth.

#### Qwen VLM option (free, Hugging Face)

Chest Xpert includes a `QwenVLClient` / `QwenVerificationEngine` for Stage B that supports:

| Backend | How it works | Cost | VRAM needed |
|---------|-------------|------|-------------|
| `hf_api` | Hugging Face Inference API at `https://api-inference.huggingface.co/models/{model_id}` | Free tier (rate-limited) | None (server-side) |
| `transformers_local` | Loads Qwen2.5-VL via `transformers` locally | Free (your hardware) | ~15 GB FP16 for 7B |

Recommended free model: `Qwen/Qwen2.5-VL-7B-Instruct` (vision-language, can inspect the X-ray image).

Set in the Streamlit sidebar (or as env vars):
- `CHESTXPERT_QWEN_BACKEND` = `hf_api` or `transformers_local`
- `CHESTXPERT_QWEN_MODEL_ID` = `Qwen/Qwen2.5-VL-7B-Instruct`
- `CHESTXPERT_HF_TOKEN` = your HF read token

The token is read only from environment variables or the password input — never hardcoded, never logged, never sent in chat.


### Stage C — LLM #2 report generation

- Runs only after Stage 0 + Stage A succeed (or verification explicitly marked unavailable).
- Separate model configuration (may differ from LLM #1).
- Produces structured report: exam details, model findings, scores, verification assessments, agreements/disagreements, uncertainty, draft impression, reviewer fields.
- Deterministic template-based fallback if LLM #2 fails.
- All reports labeled **AI-generated research draft requiring radiologist review**.

### Stage D — Human review & export

- Streamlit UI showing uploaded image, validation status, classifier scores, supported vs. unavailable labels, Grad-CAM, LLM #1 results, disagreements, editable report, stage-level + total latency, model names/versions/errors.
- Classifier output, LLM assessment, and report wording are visually distinct.
- PDF download and browser printing supported.

---

## 6. Sample-dataset smoke test

`tests/test_sample_dataset.py` runs Stage 0 + Stage A against:
- 1 real chest X-ray (bundled `sample_frontal.png`)
- Synthetic fixtures: color photograph, grayscale document, cartoon, corrupted file, empty file

Results written to `tests/sample_dataset_results.json`.

**Observed results (CPU, with CLIP enabled via torch 2.7.1):**

| Image | File validation | Modality decision | Classifier run? |
|-------|-----------------|-------------------|-----------------|
| valid_chest_xray | OK | accepted | ✅ classified |
| color_photograph | OK | rejected | skipped |
| grayscale_document | OK | rejected | skipped |
| cartoon | OK | rejected | skipped |
| corrupted_file | FAIL (.bin) | — | skipped |
| empty_file | FAIL (unreadable) | — | skipped |

**Classifier scores on the valid chest X-ray** (measured separately, CPU, ~353 ms total):

| Label | Score |
|-------|-------|
| Atelectasis | 0.8310 |
| Consolidation | 0.6365 |
| Infiltration | 0.5918 |
| Pneumothorax | 0.5166 |
| Edema | 0.5523 |
| Emphysema | 0.4737 |
| Fibrosis | 0.5109 |
| Effusion | 0.7260 |
| Pneumonia | 0.5534 |
| Pleural Thickening | 0.5071 |
| Cardiomegaly | 0.6262 |
| Nodule | 0.5159 |
| Mass | 0.5776 |
| Hernia | 0.5517 |
| Lung Lesion | 0.5433 |
| Fracture | 0.5626 |
| Lung Opacity | 0.7615 |
| Enlarged Cardiomediastinum | 0.5535 |

> These are raw outputs from a single test image — **not** evaluation metrics, not comprehensive, not clinically validated.

### Measured latency (CPU, one 224×224 image)
- Preprocessing: ~3 ms
- Model inference (forward): ~541 ms
- Stage A total: ~544 ms
- Grad-CAM: ~109 ms
- PDF report: generated successfully

Sub-500 ms was investigated (~544 ms on CPU) but is **not promised**. The Tesla V100 GPU is shared and near capacity on this server, so CPU is the reliable default.

---

## 7. Tests

21 tests pass (`pytest tests/`):
- `tests/test_core.py` (9): label mapping, all-18 support, preprocessing shape/range, file validation, report HTML/PDF, disclaimer, failure handling.
- `tests/test_pipeline.py` (12): modality decision logic, verification schema validation & disagreement handling, report with verification/modality data.

No training runs in tests.

---

## 8. How to run locally

```bash
# 1. Activate the existing environment (do NOT create a new one)
source /home/doli/anaconda3/bin/activate hct-lsr

# 2. Go to the project
cd "/home/doli/Chest Xpert"

# 3. Install dependencies (into the existing env)
pip install -r requirements.txt

# 4. (Optional) Get a free Hugging Face token for LLM Stage B
#    - Go to https://huggingface.co/settings/tokens
#    - Create a "Read" token (free)
#    - Export it (never paste it in chat or commit it):
export CHESTXPERT_HF_TOKEN="hf_your_token_here"

# 5. Run the Streamlit app
streamlit run chest_xpert/app.py

# 6. Open the URL shown (e.g. http://localhost:8502)
#    In the sidebar, configure LLM:
#    - Backend: transformers_local (recommended, runs on CPU)
#    - Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1 GB, fastest)
#    - Force CPU: enabled
#    - No HF token needed for local backend
```

**LLM model options for Stage B (all free, all on Hugging Face):**

| Model | Type | Size | RAM/VRAM | Best for |
|-------|------|------|----------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | Text-only | ~2 GB | 2 GB RAM | Best accuracy, CPU OK **(default)** |
| `Qwen/Qwen2.5-0.5B-Instruct` | Text-only | ~1 GB | 1 GB RAM | Fastest, works on CPU |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Text-only | ~1 GB | 1 GB RAM | Smallest, fastest |
| `meta-llama/Llama-3.2-1B-Instruct` | Text-only | ~2 GB | 2 GB RAM | Better accuracy |
| `microsoft/Phi-3-mini-4k-instruct` | Text-only | ~4 GB | 4 GB RAM | Best accuracy |
| `Qwen/Qwen3-VL-32B-Instruct` | Vision-Language | ~16 GB (4-bit) | 16+ GB GPU | Can see X-ray image |

**Note:** Qwen3-VL-32B requires ~16 GB VRAM in 4-bit and may not fit on shared GPUs. It also requires `bitsandbytes` which may have compatibility issues. For the shared Tesla V100 (often near capacity), use text-only models like Qwen2.5-1.5B on CPU.

**Backend options:**
- `transformers_local` — loads model locally via transformers (default, no server needed)
- `hf_api` — Hugging Face Inference API (needs `CHESTXPERT_HF_TOKEN` + internet)
- `vllm_server` — connect to self-hosted vLLM server at `CHESTXPERT_VLLM_URL`

**Optional: enable generic remote LLM** (not required; app works in fallback mode without it):
```bash
export CHESTXPERT_LLM_API_URL="https://your-llm-endpoint/v1/chat/completions"
export CHESTXPERT_LLM_API_TOKEN="your-token"
export CHESTXPERT_LLM_MODEL_ID="your-model-id"
```

**How Stage B works (single LLM):**

The LLM receives classifier scores for all 18 pathologies and returns:
1. **Verification** — per-pathology `consistent` / `inconsistent` / `uncertain` with rationale
2. **Report findings** — derived deterministically: `consistent` + score ≥ 0.5 → `supported`; `inconsistent` → `not_supported`; `uncertain` → `uncertain`

The LLM is instructed **not** to report findings based on high scores alone. Findings are only supported when the LLM assessment agrees with the classifier output.

**Run tests:**
```bash
source /home/doli/anaconda3/bin/activate hct-lsr
cd "/home/doli/Chest Xpert"
pip install pytest   # if not present
pytest tests/ -q
```

**Run sample-dataset smoke test:**
```bash
source /home/doli/anaconda3/bin/activate hct-lsr
cd "/home/doli/Chest Xpert"
python tests/test_sample_dataset.py
```

---

## 9. How to deploy for free

Chest Xpert is a Streamlit app. Free hosting options:

### Option A — Hugging Face Spaces (recommended for ML demos, free)
1. Create a free HF account: https://huggingface.co (join)
2. From the remote terminal (token never shared in chat):
   ```bash
   source /home/doli/anaconda3/bin/activate hct-lsr
   pip install huggingface_hub
   huggingface-cli login   # paste your HF token when prompted; stored locally under ~/.cache/huggingface/token
   ```
3. Create a new **Space** on HF: choose "Streamlit" SDK, set visibility to **Private** (default until you choose to publish).
4. Upload the project files (`chest_xpert/`, `requirements.txt`, `README.md`, `app.py` entrypoint).
5. In Space settings, add secrets `CHESTXPERT_LLM_API_URL` and `CHESTXPERT_LLM_API_TOKEN` if you want LLM stages (otherwise app runs in fallback mode).
6. The Space builds and hosts the app for free.

> Do **not** upload third-party pretrained weights as your own. Reference them (torchxrayvision downloads weights at runtime). The `hf_repo/README.md` and `MODEL_CARD.md` already carry the correct attribution.

### Option B — Streamlit Community Cloud (free)
1. Push the project to a public/private GitHub repo.
2. Connect to https://streamlit.io/cloud (free tier).
3. Point to `chest_xpert/app.py` as the entrypoint.
4. Add secrets in the cloud UI for LLM env vars if needed.

### Option C — Render free tier
1. Connect GitHub repo to Render.
2. Create a "Web Service" with the free tier.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run chest_xpert/app.py --server.port $PORT --server.headless true`

### Local-only deployment (always free, no internet needed after setup)
Run the app on this server and access via the displayed network URL (e.g. `http://10.203.6.228:8502`) if you're on the same network, or use SSH tunneling:
```bash
# From your local machine
ssh -L 8502:localhost:8502 doli@<server-ip>
# Then open http://localhost:8502 in your browser
```

---

## 10. Deployment checklist / what you need

- [ ] Existing Conda env `hct-lsr` activated (no new env).
- [ ] `pip install -r requirements.txt` (streamlit, reportlab, pydicom, torchxrayvision already present).
- [ ] Sample image in `samples/` (already bundled).
- [ ] (Optional) HF token for publishing; never hardcode it.
- [ ] (Optional) LLM endpoint + token for Stages B/C; app works without them.
- [ ] Decision on visibility: keep HF repo/Space **private** until you explicitly publish.
- [ ] Medical disclaimer understood: app is a research draft, not a clinical device.

---

## 11. Next action steps

1. **Review the code** in `chest_xpert/app.py` (pipeline UI) and `docs/` (methodology + architecture).
2. **Run locally** using the commands in §8 and confirm the pipeline end-to-end.
3. **Run the smoke test** (`python tests/test_sample_dataset.py`) and review `tests/sample_dataset_results.json`.
4. **Decide on LLM integration:** if you have an authorized VLM endpoint, set the env vars; otherwise Stages B/C run in deterministic fallback/unavailable mode.
5. **Decide on deployment:** choose Hugging Face Spaces (private), Streamlit Community Cloud, Render, or local-only.
6. **Prepare for publishing (optional):** when ready, log in via `huggingface-cli login` from the remote terminal, create a private Space/repo, upload files, and flip visibility to public only when you explicitly choose to.
7. **Evaluation (future):** when you have a properly licensed, expert-annotated test dataset, run `chest_xpert.evaluation.evaluate_dataset()` and record per-label AUROC/AUPRC. Do not claim results until measured.

---

## 12. Important limitations & disclaimers

- **Modality detection is not validated.** CLIP now loads successfully (torch upgraded to 2.7.1). It correctly accepted the bundled chest X-ray at 95.7% confidence and rejected color photos, documents, and cartoons. However, CLIP zero-shot is still a heuristic — not a validated clinical test. Do not claim reliable clinical modality detection.
- **No LLM configured by default.** Stages B/C are unavailable unless you provide endpoint/token via environment variables.
- **No evaluation performed.** No expert-annotated dataset was used; AUROC/AUPRC/modality metrics are not measured. The smoke test is functional only.
- **Scores are not calibrated probabilities.** Not clinical confidence.
- **Grad-CAM is an explanation, not proof of disease.**
- **Modality detection is not validated.** Qwen VLM and CLIP are heuristics — not validated clinical tests. Qwen may misclassify unusual X-ray views or non-standard images. Always review results with a radiologist.
- **Not a medical device.** Every report is an AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis.
- **No training or fine-tuning** was performed. No third-party model is misrepresented as your own.
