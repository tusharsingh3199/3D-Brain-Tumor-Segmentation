import os
import shutil
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from src.training.DiceLoss import Loss, Dice, Dice_NCR, Dice_ED, Dice_ET

MODEL_PATH = "3D_UNet.keras"          # change if using Swin_UNETR.keras
PATCH = (128, 128, 128)
STRIDE = (64, 64, 64)

app = FastAPI(title="Brain Tumor Segmentation API")

# Load model once at startup
model = tf.keras.models.load_model(
    r"C:\Users\dell\Documents\Programming Projects\Python\3D Brain tumor Segmentation\Models\3D_UNet.keras",
    custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR,
                     "Dice_ED": Dice_ED, "Dice_ET": Dice_ET},
)


def load_volume(path):
    v = nib.load(path).get_fdata().astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v


def sliding_window_predict(img):
    """Same logic as mri_results.py, wrapped into a function."""
    H, W, D, _ = img.shape
    prediction = np.zeros((H, W, D, 4), dtype=np.float32)
    count_map = np.zeros((H, W, D, 1), dtype=np.float32)

    xs = list(range(0, H - PATCH[0] + 1, STRIDE[0])) + [H - PATCH[0]]
    ys = list(range(0, W - PATCH[1] + 1, STRIDE[1])) + [W - PATCH[1]]
    zs = list(range(0, D - PATCH[2] + 1, STRIDE[2])) + [D - PATCH[2]]
    xs, ys, zs = sorted(set(xs)), sorted(set(ys)), sorted(set(zs))

    for x in xs:
        for y in ys:
            for z in zs:
                patch = img[x:x+PATCH[0], y:y+PATCH[1], z:z+PATCH[2], :]
                patch = np.expand_dims(patch, axis=0)
                pred = model.predict(patch, verbose=0)[0]
                prediction[x:x+PATCH[0], y:y+PATCH[1], z:z+PATCH[2]] += pred
                count_map[x:x+PATCH[0], y:y+PATCH[1], z:z+PATCH[2]] += 1

    prediction /= count_map
    return np.argmax(prediction, axis=-1).astype(np.uint8)


@app.post("/segment")
async def segment(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    tmp_dir = tempfile.mkdtemp()
    files = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    paths = {}

    for name, upload in files.items():
        path = os.path.join(tmp_dir, f"{name}.nii")
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        paths[name] = path

    # Use the affine of the first scan for the output NIfTI
    ref_nii = nib.load(paths["t1"])
    volumes = [load_volume(paths[m]) for m in ["t1", "t1ce", "t2", "flair"]]
    img = np.stack(volumes, axis=-1)

    seg = sliding_window_predict(img)
    seg[seg == 3] = 4  # map back to BraTS label convention (0,1,2,4)

    out_path = os.path.join(tmp_dir, "seg.nii.gz")
    nib.save(nib.Nifti1Image(seg, ref_nii.affine), out_path)

    return FileResponse(out_path, media_type="application/gzip", filename="seg.nii.gz")


@app.get("/")
def health():
    return {"status": "ok"}

