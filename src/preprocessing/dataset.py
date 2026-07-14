import os
import nibabel as nib
import numpy as np
import tensorflow as tf


def load_patient(patient_dir):
    patient_dir = patient_dir.numpy().decode()
    images = []
    for modality in ["t1", "t1ce", "t2", "flair"]:
        v = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii")).get_fdata().astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        images.append(v)

    image = np.stack(images, axis=-1).astype(np.float32)
    mask = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii")).get_fdata().astype(np.int32)
    mask[mask == 4] = 3

    patch_size = (128, 128, 128)
    tumor = np.argwhere(mask > 0)

    if len(tumor) > 0:
        cx, cy, cz = tumor[np.random.randint(len(tumor))]
        x = np.clip(cx - patch_size[0] // 2, 0, image.shape[0] - patch_size[0])
        y = np.clip(cy - patch_size[1] // 2, 0, image.shape[1] - patch_size[1])
        z = np.clip(cz - patch_size[2] // 2, 0, image.shape[2] - patch_size[2])
    else:
        x = np.random.randint(0, image.shape[0] - patch_size[0] + 1)
        y = np.random.randint(0, image.shape[1] - patch_size[1] + 1)
        z = np.random.randint(0, image.shape[2] - patch_size[2] + 1)

    image = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2], :]
    mask = mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]

    return image, mask


def tf_load_patient(patient_dir):
    image, mask = tf.py_function(load_patient, [patient_dir], [tf.float32, tf.int32])
    image.set_shape((128, 128, 128, 4))
    mask.set_shape((128, 128, 128))
    return image, mask

