import os
import shutil
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from src.training.DiceLoss import *
from configs.config import *
from src.training.mri_results import sliding_window_predict

app = FastAPI(title="Brain Tumor Segmentation API")

# Load model once at startup
UNET = tf.keras.models.load_model(DATA_PATH + r"\Models\3D_UNet.keras",
    custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET},
)

SwinUNETR = tf.keras.models.load_model(DATA_PATH + r"\Models\Swin_UNETR.keras",
    custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET},
)


@app.post("/segment")
async def segment(t1: UploadFile = File(...), t1ce: UploadFile = File(...),
                  t2: UploadFile = File(...), flair: UploadFile = File(...),):

    tmp_dir = tempfile.mkdtemp()
    files = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    paths = {}

    for name, upload in files.items():
        path = os.path.join(tmp_dir, f"{name}.nii")
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        paths[name] = path

    ref_nii = nib.load(paths["t1"])
    images = []

    for m in MODALITIES:
        volume = nib.load(paths[m]).get_fdata().astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        images.append(volume)

    img = np.stack(images, axis=-1)
    seg = sliding_window_predict(img, UNET)
    seg[seg == 3] = 4

    out_path = os.path.join(tmp_dir, "seg.nii.gz")
    nib.save(nib.Nifti1Image(seg, ref_nii.affine), out_path)

    return FileResponse(out_path, media_type="application/gzip", filename="seg.nii.gz")


@app.get("/")
def health():
    return {"status": "ok"}

