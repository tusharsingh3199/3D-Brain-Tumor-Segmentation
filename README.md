# 3D Brain Tumor Segmentation using 3D UNet and Swin UNETR

End-to-end deep learning pipeline for automated brain tumor segmentation from multi-modal MRI scans using **3D U-Net** and **Swin-UNETR**.
Built with TensorFlow, FastAPI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

Brain tumor segmentation is the task of automatically delineating tumor sub-regions in 3D MRI volumes so that clinicians and researchers can analyze tumor size, shape, and location without manually tracing every slice. This project implements a complete, reproducible pipeline for that task: data loading and preprocessing, two independent 3D segmentation architectures, a combined loss/metric suite tailored to class imbalance, a training script, and a deployable inference stack (a FastAPI backend plus a Streamlit front end) so trained models can be used interactively rather than only from a notebook.

Two models are provided so their trade-offs can be compared directly on the same data and evaluation code:

- A convolutional **3D U-Net**, used as a strong, well-understood baseline.
- A **Swin-UNETR**, which replaces the convolutional encoder with a 3D shifted-window transformer to capture longer-range spatial context.

Both models consume four co-registered MRI modalities per patient (T1, T1ce, T2, FLAIR) stacked as input channels, and predict a voxel-wise segmentation mask distinguishing background from the tumor sub-regions described below.

## Key Features

- Two interchangeable 3D segmentation architectures (3D U-Net and Swin-UNETR) trained and evaluated with the same pipeline.
- Custom combined Dice and Cross-Entropy loss with class weighting to handle the heavy background/tumor voxel imbalance typical of brain MRI.
- Per-class Dice metrics tracked during training for necrotic core, edema, and enhancing tumor regions.
- Patch-based training and sliding-window inference so full-resolution volumes can be processed without exceeding memory limits.
- REST API (FastAPI) that loads both trained models once at startup and lets the caller choose which model to run per request.
- Interactive Streamlit UI for uploading scans, running inference, scrolling through slice-by-slice overlays, and downloading the resulting segmentation mask.
- Trained model weights are pulled automatically from the Hugging Face Hub if not already present locally, so inference can be run without retraining from scratch.
- Single-command orchestration (`main.py`) that trains missing models and launches both services.

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | TensorFlow / Keras |
| Architectures | 3D U-Net, Swin-UNETR (3D Swin Transformer encoder + convolutional decoder) |
| Data handling | NiBabel (NIfTI I/O), NumPy, SciPy |
| Backend / inference API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Model distribution | Hugging Face Hub |
| Experimentation | Jupyter Notebook |

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
Multi-modal MRI NIfTI files (t1, t1ce, t2, flair, seg)
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

Both `Models/3D_UNet.keras` and `Models/Swin_UNETR.keras` are loaded at API startup. The `/segment` endpoint takes a `model` field (`"UNET"` or `"SwinUNETR"`) so you can pick which one to run inference with per request, with no restart or code change needed to switch models.

## Dataset

This project trains on a multi-modal brain MRI dataset organized by patient, where each patient folder contains four co-registered structural MRI modalities plus a ground-truth segmentation mask:

- **T1** — native T1-weighted scan.
- **T1ce** — T1-weighted scan with contrast enhancement, which highlights the active tumor boundary.
- **T2** — T2-weighted scan, sensitive to edema and fluid.
- **FLAIR** — fluid-attenuated inversion recovery, which suppresses cerebrospinal fluid signal to make lesions more visible.
- **seg** — the ground-truth voxel-wise label map used as the training target.

Patients are typically grouped into high-grade and low-grade glioma cohorts (HGG/LGG). Each of the four modalities is intensity-normalized independently and stacked along the channel axis, giving a 4-channel 3D input volume per patient. During training, volumes are cropped into fixed-size 3D patches (128³ voxels by default, configurable in `configs/config.py`) to keep GPU memory usage manageable; at inference time the same patch size is applied via a sliding window across the full volume so the whole scan gets segmented.

Labels are remapped from their raw encoding to a compact set of contiguous class indices (0, 1, 2, 3) before training, and mapped back to the original label convention when writing out predictions.

## Setup

### 1. Clone the repository and install dependencies

```bash
git clone <repository-url>
cd brain_tumor_segmentation
pip install -r requirements.txt
```

### 2. Obtain the dataset

The dataset can be downloaded automatically via `src/data/loader.py` (using the Kaggle API), or placed manually at:

```
Data/MICCAI_BraTS_2019_Data_Training/HGG/
Data/MICCAI_BraTS_2019_Data_Training/LGG/
```

Each patient subfolder must contain the four MRI modality files and the ground-truth segmentation file, following the naming convention used by `src/data/dataset.py`.

### 3. Trained model weights (optional)

Trained weights do not need to be produced locally to try inference: `scripts/api.py` automatically downloads `3D_UNet.keras` and `Swin_UNETR.keras` from the configured Hugging Face Hub repository into `Models/` the first time the API is started, if they are not already present.

## Usage

### Run everything with one command

```bash
python main.py
```

This will:

1.  Train the models (`scripts/train.py`) if they don't already exist in `Models/`.
2.  Start the FastAPI server on `http://localhost:8000`.
3.  Start the Streamlit UI on `http://localhost:8501`.

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

1.  Open `http://localhost:8501`.
2.  Select which model to run: **UNET** or **SwinUNETR**.
3.  Upload the 4 MRI modalities: T1, T1ce, T2, FLAIR (`.nii` / `.nii.gz`).
4.  Click **Run Segmentation** — the API runs sliding-window inference with the selected model and returns the predicted mask.
5.  Use the slider to scroll through slices; the top row shows the raw scans, the bottom row shows the scans with the predicted tumor regions overlaid.
6.  Click **Download seg.nii.gz** to save the segmentation mask.

### Using the API directly

Both trained models (`3D_UNet.keras` and `Swin_UNETR.keras`) are loaded at startup. Pick which one to run inference with via the `model` form field — `"UNET"` or `"SwinUNETR"`.

```bash
curl -X POST http://localhost:8000/segment \
  -F "model=UNET" \
  -F "t1=@patient_t1.nii" \
  -F "t1ce=@patient_t1ce.nii" \
  -F "t2=@patient_t2.nii" \
  -F "flair=@patient_flair.nii" \
  -o seg.nii.gz
```

Health check:

```bash
curl http://localhost:8000/
```

## Model Architectures

-   **3D U-Net** (`src/models/model_3D_UNet.py`) — encoder-decoder with skip connections, Conv3D + BatchNorm blocks, trained as the baseline.
-   **Swin-UNETR** (`src/models/model_SwinUNETR.py`) — Swin Transformer 3D encoder (windowed self-attention with shifted windows) paired with a convolutional decoder, trained as the advanced model.

Both are trained with a combined Dice + Cross-Entropy loss (`src/training/DiceLoss.py`), with per-class Dice tracked for:

-   **NCR** (Necrotic/Non-enhancing Core)
-   **ED** (Peritumoral Edema)
-   **ET** (Enhancing Tumor)

Class weighting is applied in the loss function to counteract the large imbalance between background voxels and the (comparatively small) tumor sub-region voxels.

## Training Configuration

Key parameters, defined centrally in `configs/config.py`, include:

| Parameter | Description |
|---|---|
| `VOLUME_SIZE` | Raw input volume dimensions before patching. |
| `PATCH_SIZE` | Size of the 3D patches extracted for training and inference. |
| `STRIDE` | Step size used when sliding the inference window across a full volume. |
| `CLASS_WEIGHTS` | Per-class weights applied in the loss function to address class imbalance. |
| `EPOCHS` | Number of training epochs. |
| `MODELS` | Which architecture(s) to train (`unet`, `swin_unetr`). |

Adjust these values to change patch size, training duration, or which model(s) get trained without touching the training script itself.

## Results

Dice scores from evaluating on the held-out test split (`test_dir` in `scripts/train.py`). Replace the placeholder values below with your own numbers after training — printed by `UNET_Model.evaluate(test_dataset)` / `Swin_UNETR.evaluate(test_dataset)`, or computed per-patient via `src/training/mri_results.py`.

| Class | 3D U-Net | Swin-UNETR |
|---|---|---|
| Whole Tumor (WT) | 0.81 | 0.74 |
| Necrotic / Non-enhancing (NCR) | 0.70 | 0.55 |
| Peritumoral Edema (ED) | 0.80 | 0.71 |
| Enhancing Tumor (ET) | 0.74 | 0.70 |
| **Mean Dice** | **0.00** | **0.00** |

> WT, TC (Tumor Core = NCR + ET), and ET are the standard evaluation regions used in tumor segmentation benchmarks. NCR/ED/ET above are the raw per-class scores from `DiceLoss.py`; combine them (e.g. NCR + ET voxels for TC) if you want region-level scores instead of per-label scores.

## Demo

[![Watch the demo](https://www.youtube.com/watch?v=UjHknx9umLM/0.jpg)](https://www.youtube.com/watch?v=UjHknx9umLM)

A short walkthrough of the Streamlit UI (select model, upload the four MRI modalities, run segmentation, scroll through the slice viewer, download the mask) can be added here once recorded, using one of the embedding options above.

## Limitations

- Trained on a fixed set of four MRI modalities (T1, T1ce, T2, FLAIR); scans missing one or more modalities cannot be processed without modification.
- Sliding-window inference assumes co-registered volumes of consistent orientation and spacing; scans that differ significantly from the training distribution may degrade segmentation quality.
- Model performance depends on the size and diversity of the training cohort and has not been validated across multiple scanners, institutions, or patient populations.

## Disclaimer

This project is for research and educational purposes only and is not intended for clinical use. Segmentation outputs must not be used for diagnosis or treatment decisions without review by qualified medical professionals.

## License

Released under the MIT License.
