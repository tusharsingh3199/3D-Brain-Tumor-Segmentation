import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from ipywidgets import interact, IntSlider


def MRI_Results(patients_dir, model):
    patient_dir = random.choice(patients_dir)
    images = []
    patient_id = os.path.basename(patient_dir)

    for modality in ["t1", "t1ce", "t2", "flair"]:
        volume = nib.load(os.path.join(patient_dir, f"{patient_id}_{modality}.nii")).get_fdata().astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        images.append(volume)

    image = np.stack(images, axis=-1)
    mask = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii")).get_fdata().astype(np.int32)
    mask[mask == 4] = 3

    H, W, D, C = (240, 240, 155, 4)
    patch_size = (128, 128, 128)
    stride = (64, 64, 64)

    prediction = np.zeros((H, W, D, 4), dtype=np.float32)
    count_map = np.zeros((H, W, D, 1), dtype=np.float32)

    xs = list(range(0, H - patch_size[0] + 1, stride[0]))
    ys = list(range(0, W - patch_size[1] + 1, stride[1]))
    zs = list(range(0, D - patch_size[2] + 1, stride[2]))

    if xs[-1] != H - patch_size[0]:
        xs.append(H - patch_size[0])
    if ys[-1] != W - patch_size[1]:
        ys.append(W - patch_size[1])
    if zs[-1] != D - patch_size[2]:
        zs.append(D - patch_size[2])

    for x in xs:
        for y in ys:
            for z in zs:
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2], : ]
                patch = np.expand_dims(patch, axis=0)
                pred = model.predict(patch, verbose=0)[0]

                prediction[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += pred
                count_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1

    prediction /= count_map
    prediction = np.argmax(prediction, axis=-1)

    def view_slice(z):
        plt.figure(figsize=(8, 6))
        titles = ["T1","T1CE","T2","FLAIR"]

        for i in range(4):

            plt.subplot(3,4,i+1)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.title(titles[i])
            plt.axis("off")

            plt.subplot(3,4,5+i)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.imshow(np.ma.masked_where(mask[:,:,z]==0, mask[:,:,z]),
                       cmap="jet", alpha=0.6, vmin=0, vmax=3)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(3,4,9+i)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.imshow(np.ma.masked_where(prediction[:,:,z]==0, prediction[:,:,z]),
                       cmap="jet", alpha=0.6, vmin=0, vmax=3)
            plt.title("Prediction")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    interact(view_slice, z=IntSlider(min=0, max=154, value=64))