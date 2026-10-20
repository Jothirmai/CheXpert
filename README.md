# Chest Xpert

Chest X-ray multi-label findings inference using existing pre-trained models —
**no training, no fine-tuning**.

## Project

`/home/doli/Chest Xpert`

## Environment (reused, not created)

- Conda env: `/home/doli/anaconda3/envs/hct-lsr` (Python 3.10.21)
- torch 2.5.1+cu118, torchvision, torchxrayvision 1.5.4, numpy, scikit-learn, etc.
- Hardware: Tesla V100 32 GB (shared, often near capacity) + Quadro K620 2 GB;
  ~156 GB RAM; ~1.9 TB free disk.

## Selected pre-trained model

torchxrayvision `DenseNet-121` (`densenet121-res224-all`) — supports **all 18**
requested chest X-ray labels. Weights created by torchxrayvision / CheXpert
researchers, **not** by this project. Referenced at runtime from
`~/.torchxrayvision/models_data/` (already cached on this server).

## 18 labels

All supported: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema,
Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly,
Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged
Cardiomediastinum.

## Usage

```bash
source /home/doli/anaconda3/bin/activate hct-lsr
pip install -r requirements.txt
streamlit run chest_xpert/app.py
```

## Disclaimer

AI-generated research draft requiring review by a qualified radiologist. Not a
final clinical diagnosis. Scores are not calibrated probabilities.
