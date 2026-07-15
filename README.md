# 3D Brain Tumor Segmentation (BraTS) — TensorFlow

End-to-end pipeline for 3D brain tumor segmentation on the BraTS 2021 dataset,
progressing from a 2D U-Net baseline to a Swin-UNETR transformer, with
uncertainty estimation, explainability, and a deployable clinical API + UI.

## Project Structure

```
brain_tumor_segmentation/
├── configs/                    # Single source of truth for all hyperparameters
│   └── config.py
│
├── src/
│   ├── data/                   # Loading, preprocessing, augmentation, tf.data pipelines
│   │   ├── loader.py           #   NIfTI + DICOM I/O
│   │   ├── preprocessor.py     #   normalize, remap labels, crop, pad, one-hot
│   │   ├── augmentation.py     #   2D and 3D augmentation
│   │   └── dataset.py          #   tf.data pipeline builders (2D and 3D)
│   │
│   ├── models/                 # Model architectures
│   │   ├── unet2d.py           #   2D U-Net (Phase 2 baseline)
│   │   ├── unet3d.py           #   3D U-Net (Phase 3)
│   │   └── swin_unetr.py        #   Swin Transformer + CNN decoder (Advanced 1)
│   │
│   ├── training/                # Losses, metrics, training loop helpers
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── trainer.py           #   callbacks, optimizers, save/load
│   │
│   ├── inference/               # Sliding window inference
│   │   └── sliding_window.py    #   Gaussian-weighted patch stitching
│   │
│   ├── explainability/          # Advanced 2 — uncertainty & XAI
│   │   ├── mc_dropout.py        #   Monte Carlo Dropout uncertainty maps
│   │   ├── gradcam.py           #   Grad-CAM 3D
│   │   ├── integrated_gradients.py
│   │   └── shap_explainer.py    #   GradientSHAP
│   │
│   ├── deployment/               # Advanced 3 — clinical API
│   │   ├── model_manager.py      #   singleton model loader
│   │   ├── segmentation_engine.py
│   │   ├── analysis.py           #   volumes, Dice, BraTS regions
│   │   ├── report_generator.py   #   HTML / JSON reports
│   │   ├── tflite_converter.py
│   │   ├── pipeline.py           #   end-to-end orchestration
│   │   └── api.py                #   FastAPI app
│   │
│   └── visualization/            # Plots and Gradio UI
│       ├── plots.py
│       └── gradio_app.py         #   Advanced 4 — web UI + DICOM
│
├── scripts/                       # CLI entry points (run these)
│   ├── 01_preprocess.py           #   Phase 1
│   ├── 02_train_unet2d.py         #   Phase 2
│   ├── 03_train_unet3d.py         #   Phase 3
│   ├── 04_train_swin_unetr.py     #   Advanced 1
│   ├── 05_explainability.py       #   Advanced 2
│   ├── 06_run_inference.py        #   Advanced 3 (pipeline / benchmark)
│   ├── 07_convert_tflite.py       #   TFLite quantization
│   ├── 08_dicom_tools.py          #   DICOM convert / segment
│   ├── run_api.py                 #   FastAPI server
│   └── run_ui.py                  #   Gradio UI
│
├── tests/                          # pytest unit tests
├── nginx/nginx.conf                # Reverse proxy config
├── .github/workflows/ci_cd.yml     # Lint → test → Docker → GCP Cloud Run
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt                # Core (training)
├── requirements-deploy.txt         # + API, UI, DICOM
└── requirements-dev.txt            # + testing/linting
```

## Data Flow / Pipeline

```
BraTS NIfTI files
   │
   ▼
src/data/loader.py + preprocessor.py   (normalize, remap labels 0,1,2,4 → 0,1,2,3)
   │
   ├──► 2D slices ──► src/models/unet2d.py     ──► saved_model/unet2d_final
   │
   └──► 3D patches ─► src/models/unet3d.py     ──► saved_model/unet3d_final
                  └──► src/models/swin_unetr.py ──► saved_model/swin_unetr_final
                                │
                                ▼
                  src/inference/sliding_window.py (Gaussian-weighted stitching)
                                │
        ┌───────────────────────┼──────────────────────────┐
        ▼                        ▼                          ▼
src/explainability/    src/deployment/pipeline.py    src/visualization/gradio_app.py
(MC Dropout, Grad-CAM,  (volumes, Dice, HTML/JSON      (web UI, DICOM I/O,
 IG, SHAP)               reports, FastAPI)              multi-plane viewer)
```

`saved_model/unet3d_final` (or `swin_unetr_final`, copied to the same path) is
the single artifact that every downstream component — explainability, API,
UI — reads from. Swapping in a better model requires no other code changes.

## Setup

```bash
# Core dependencies (training only)
pip install -r requirements.txt

# Full deployment stack (API, UI, DICOM)
pip install -r requirements-deploy.txt

# Development (testing, linting)
pip install -r requirements-dev.txt
```

Download the BraTS 2021 dataset and place it at:
```
data/BraTS2021_Training_Data/BraTS2021_XXXXX/
```

## Usage

### 1. Preprocess (Phase 1)
```bash
make preprocess
```

### 2. Train 2D U-Net baseline (Phase 2)
```bash
make train-2d
```

### 3. Train 3D U-Net (Phase 3)
```bash
make train-3d
```

### 4. Train Swin-UNETR transformer (Advanced 1)
```bash
make train-swin
# To use it everywhere downstream:
cp -r saved_model/swin_unetr_final saved_model/unet3d_final
```

### 5. Uncertainty & Explainability (Advanced 2)
```bash
make explain
# outputs/ uncertainty_outputs/uncertainty_*.png, gradcam_*.png,
#          integrated_grads_*.png, shap_*.png
```

### 6. Clinical Inference Pipeline (Advanced 3)
```bash
make infer                 # single patient → HTML + JSON + NIfTI report
make benchmark              # BraTS Dice benchmark over N patients
make convert-tflite          # INT8-quantized TFLite export
```

### 7. API + UI (Advanced 4)
```bash
make api     # FastAPI on :8000
make ui      # Gradio on :7860
```

### DICOM
```bash
python scripts/08_dicom_tools.py convert --dicom-dir /path/to/dcm --out outputs/scan.nii.gz
python scripts/08_dicom_tools.py segment --dicom-dir /path/to/dcm --out-dir outputs/seg_dicom
```

## Docker

```bash
make build     # build image
make run       # API:8000, UI:7860, Nginx:80
make logs
make stop
```

## Testing

```bash
make test      # pytest with coverage
make lint      # flake8 + black
```

## Model Performance Targets (BraTS 2021)

| Region              | Target Dice |
|---------------------|-------------|
| Whole Tumor (WT)    | ≥ 0.88      |
| Tumor Core (TC)     | ≥ 0.78      |
| Enhancing Tumor (ET)| ≥ 0.70      |

## Disclaimer

This project is for research and educational purposes only and is not
intended for clinical use. Segmentation outputs must not be used for
diagnosis or treatment decisions without review by qualified medical
professionals.
