import os
import secrets
import nibabel as nib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from configs.config import *


def EDA(patients_dir):

    patient_dir = secrets.choice(patients_dir)
    img = []
    seg = []
    for modality in MODALITIES:
        v = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii")).get_fdata().astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        img.append(v)

    seg = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii")).get_fdata().astype(np.int32)

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Brain Tumor - Exploratory Data Analysis")
    plt.subplots_adjust(bottom=0.15)
    imgs1, imgs2, masks = [], [], []
    z = 75

    for i in range(4):
        imgs1.append(ax[0, i].imshow(img[i][:, :, z], cmap="gray"))
        imgs2.append(ax[1, i].imshow(img[i][:, :, z], cmap="gray"))
        masks.append(ax[1, i].imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]), cmap="jet", alpha=0.7, vmin=1, vmax=4))

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), "Slice", 0, img[0].shape[2] - 1, valinit=z, valstep=1)

    def update(val):
        z = int(slider.val)
        for i in range(4):
            imgs1[i].set_data(img[i][:, :, z])
            imgs2[i].set_data(img[i][:, :, z])
            masks[i].set_data(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
