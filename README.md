# 3D Brain Tumor Segmentation (BraTS) — TensorFlow

End-to-end pipeline for 3D brain tumor segmentation on the BraTS 2019 dataset, progressing from a 3D U-Net baseline to a Swin-UNETR transformer.

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
│   ├── training/                           
│   │   ├── DiceLoss.py                     #   Class-wise Dice Score, and Loss function.
│   │   ├── mri_results.py                  #   Real and Predicted results from the Model.
│   │   └── plots.py                        #   Model training plots.
│   │
│   └── deployment/               
│       ├── model_manager.py      
│       └── api.py                
│
├── scripts/                       
│   ├── train.py                            # Model training script.
│
├── Data/                       
│   ├── archive.zip                         #   Zip MRI dataset
│   └── MICCAI_BraTS_2019_Data_Training     #   Patients file with t1, t2, t1ce, flair, seg files.
│       ├── HGG                             
│       ├── LGG
│
├── Experiment Notebooks/                       
│   └─ Segmentation_Models.ipyng           #   Notebook for experiments.
│
├── Models/                       
│   ├── 3D_UNet.keras                       #   3D UNet model weight file.
│   └── Swin_UNETR.keras                    #   Swin UNETR model weight file. 
│
├── .gitignore
├── main.py
├── README.md
├── docker-compose.yml
└── requirements.txt            
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

`saved_model/unet3d_final` (or `swin_unetr_final`, copied to the same path) is the single artifact that every downstream component — explainability, API, UI — reads from. Swapping in a better model requires no other code changes.

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

Region

Target Dice

Whole Tumor (WT)

≥ 0.88

Tumor Core (TC)

≥ 0.78

Enhancing Tumor (ET)

≥ 0.70

## Disclaimer

This project is for research and educational purposes only and is not intended for clinical use. Segmentation outputs must not be used for diagnosis or treatment decisions without review by qualified medical professionals.