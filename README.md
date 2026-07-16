# 3D Brain Tumor Segmentation (BraTS) — TensorFlow

End-to-end pipeline for 3D brain tumor segmentation on the BraTS 2019 dataset,
progressing from a 3D U-Net baseline to a Swin-UNETR transformer, with a
FastAPI backend and a Streamlit web UI for uploading MRI scans and viewing
segmentation results.

## Project Structure

```
brain_tumor_segmentation/
├── configs/
│   └── config.py                           #   All parameters for the project.
│
├── src/
│   ├── data/
│   │   ├── loader.py                       #   Download data and gives patient directory list.
│   │   ├── EDA.py                          #   Brain tumor analysis.
│   │   └── dataset.py                      #   Preparing dataset for the models.
│   │
│   ├── models/
│   │   ├── model_3D_UNet.py                #   3D U-Net Model.
│   │   └── model_SwinUNETR.py              #   Swin UNETR Model.
│   │
│   └── training/
│       ├── DiceLoss.py                     #   Class-wise Dice Score, and Loss function.
│       ├── mri_results.py                  #   Real and Predicted results from the Model.
│       └── plots.py                        #   Model training plots.
│
├── scripts/
│   ├── train.py                            #   Model training script.
│   ├── api.py                              #   FastAPI server (segmentation endpoint).
│   └── streamlit_app.py                    #   Streamlit UI (upload, view, download).
│
├── Data/
│   ├── archive.zip                         #   Zip MRI dataset.
│   └── MICCAI_BraTS_2019_Data_Training     #   Patient folders with t1, t2, t1ce, flair, seg files.
│       ├── HGG
│       └── LGG
│
├── Experiment Notebooks/
│   └── Segmentation_Models.ipynb           #   Notebook for experiments.
│
├── Models/
│   ├── 3D_UNet.keras                       #   Trained 3D U-Net weights.
│   └── Swin_UNETR.keras                    #   Trained Swin-UNETR weights.
│
├── .gitignore
├── main.py                                 #   Runs training → API → UI in one command.
├── README.md
└── requirements.txt
```

## Data Flow / Pipeline

```
BraTS 2019 NIfTI files (t1, t1ce, t2, flair, seg)
   │
   ▼
src/data/loader.py + dataset.py     (normalize, remap labels 0,1,2,4 → 0,1,2,3, 128³ patches)
   │
   ▼
src/models/model_3D_UNet.py  or  model_SwinUNETR.py
   │
   ▼
saved model → Models/3D_UNet.keras (or Swin_UNETR.keras)
   │
   ▼
scripts/api.py  ── sliding-window inference (Gaussian-free average, 128³ patch, 64³ stride)
   │
   ▼
scripts/streamlit_app.py  ── upload scans, view slice-by-slice overlay, download seg.nii.gz
```

Both `Models/3D_UNet.keras` and `Models/Swin_UNETR.keras` are loaded at API
startup. The `/segment` endpoint takes a `model` field (`"UNET"` or
`"SwinUNETR"`) so you can pick which one to run inference with per request —
no restart or code change needed to switch models.

## Setup

```bash
pip install -r requirements.txt
```

Download the BraTS 2019 dataset (handled automatically via `src/data/loader.py`
using the Kaggle API) or place it manually at:
```
Data/MICCAI_BraTS_2019_Data_Training/HGG/
Data/MICCAI_BraTS_2019_Data_Training/LGG/
```

## Usage

### Run everything with one command
```bash
python main.py
```
This will:
1. Train the models (`scripts/train.py`) if they don't already exist in `Models/`.
2. Start the FastAPI server on `http://localhost:8000`.
3. Start the Streamlit UI on `http://localhost:8501`.

### Or run each step manually

**1. Train the models**
```bash
python scripts/train.py
```

**2. Start the FastAPI server** (from the project root)
```bash
uvicorn scripts.api:app --port 8000
```

**3. Start the Streamlit UI**
```bash
streamlit run scripts/streamlit_app.py
```

### Using the web UI
1. Open `http://localhost:8501`.
2. Select which model to run: **UNET** or **SwinUNETR**.
3. Upload the 4 MRI modalities: T1, T1ce, T2, FLAIR (`.nii` / `.nii.gz`).
4. Click **Run Segmentation** — the API runs sliding-window inference with
   the selected model and returns the predicted mask.
5. Use the slider to scroll through slices; the top row shows the raw scans,
   the bottom row shows the scans with the predicted tumor regions overlaid.
6. Click **Download seg.nii.gz** to save the segmentation mask.

### Using the API directly
Both trained models (`3D_UNet.keras` and `Swin_UNETR.keras`) are loaded at
startup. Pick which one to run inference with via the `model` form field —
`"UNET"` or `"SwinUNETR"`.

```bash
curl -X POST http://localhost:8000/segment \
  -F "model=UNET" \
  -F "t1=@patient_t1.nii" \
  -F "t1ce=@patient_t1ce.nii" \
  -F "t2=@patient_t2.nii" \
  -F "flair=@patient_flair.nii" \
  -o seg.nii.gz
```

## Model Architectures

- **3D U-Net** (`src/models/model_3D_UNet.py`) — encoder-decoder with skip
  connections, Conv3D + BatchNorm blocks, trained as the baseline.
- **Swin-UNETR** (`src/models/model_SwinUNETR.py`) — Swin Transformer 3D
  encoder (windowed self-attention with shifted windows) paired with a
  convolutional decoder, trained as the advanced model.

Both are trained with a combined Dice + Cross-Entropy loss
(`src/training/DiceLoss.py`), with per-class Dice tracked for:
- **NCR** (Necrotic/Non-enhancing Core)
- **ED** (Peritumoral Edema)
- **ET** (Enhancing Tumor)

## Results

Dice scores from evaluating on the held-out test split (`test_dir` in
`scripts/train.py`). Replace the placeholder values below with your own
numbers after training — printed by `UNET_Model.evaluate(test_dataset)` /
`Swin_UNETR.evaluate(test_dataset)`, or computed per-patient via
`src/training/mri_results.py`.

| Class                         | 3D U-Net | Swin-UNETR |
|--------------------------------|:--------:|:----------:|
| Whole Tumor (WT)               |   0.00   |    0.00    |
| Necrotic / Non-enhancing (NCR) |   0.00   |    0.00    |
| Peritumoral Edema (ED)         |   0.00   |    0.00    |
| Enhancing Tumor (ET)           |   0.00   |    0.00    |
| **Mean Dice**                  | **0.00** |  **0.00**  |

> WT, TC (Tumor Core = NCR + ET), and ET are the standard BraTS evaluation
> regions. NCR/ED/ET above are the raw per-class scores from `DiceLoss.py`;
> combine them (e.g. NCR + ET voxels for TC) if you want official BraTS-style
> region scores instead of per-label scores.

### Target Dice (BraTS 2021 reference benchmarks)

| Region               | Target Dice |
|----------------------|-------------|
| Whole Tumor (WT)     | ≥ 0.88      |
| Tumor Core (TC)      | ≥ 0.78      |
| Enhancing Tumor (ET) | ≥ 0.70      |

These are commonly cited reference targets for well-tuned BraTS models —
useful as a rough benchmark, not a guarantee, since this project trains on
BraTS 2019 data with a smaller/simpler pipeline.

## Disclaimer

This project is for research and educational purposes only and is not
intended for clinical use. Segmentation outputs must not be used for
diagnosis or treatment decisions without review by qualified medical
professionals.
