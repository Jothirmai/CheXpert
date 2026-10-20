# Chest Xpert

Chest X-ray multi-label findings inference using existing pre-trained models — no training, no fine-tuning.

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

AI-generated research draft requiring review by a qualified radiologist. Not a final clinical diagnosis. Prediction scores are not calibrated probabilities or clinical confidence.
