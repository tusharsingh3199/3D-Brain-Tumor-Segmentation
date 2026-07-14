import os
import secrets
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
import numpy as np
import tensorflow as tf


def MRI_Results(patients_dir, model):
    patient_dir = secrets.choice(patients_dir)
    images = []
    patient_id = os.path.basename(patient_dir)

    for modality in ["t1", "t1ce", "t2", "flair"]:
        volume = nib.load(os.path.join(patient_dir, f"{patient_id}_{modality}.nii")).get_fdata().astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        images.append(volume)

    img = np.stack(images, axis=-1)
    seg = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii")).get_fdata().astype(np.int32)
    seg[seg == 4] = 3

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
                patch = img[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2], : ]
                patch = np.expand_dims(patch, axis=0)
                pred = model.predict(patch, verbose=0)[0]

                prediction[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += pred
                count_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1

    prediction /= count_map
    prediction = np.argmax(prediction, axis=-1)

    fig, ax = plt.subplots(3, 4, figsize=(12, 6))
    fig.canvas.manager.set_window_title("MRI Brain Tumor Prediction Results")
    plt.subplots_adjust(bottom=0.15)
    imgs1, imgs2, imgs3, mask1, mask2 = [], [], [], [], []
    z = 75

    for i in range(4):
        imgs1.append(ax[0, i].imshow(img[i][:, :, z], cmap="gray"))
        imgs2.append(ax[1, i].imshow(img[i][:, :, z], cmap="gray"))
        mask1.append(ax[1, i].imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]), cmap="jet", alpha=0.7, vmin=1, vmax=4))

        imgs3.append(ax[1, i].imshow(img[i][:, :, z], cmap="gray"))
        mask2.append(ax[1, i].imshow(np.ma.masked_where(prediction[:, :, z] == 0, prediction[:, :, z]), cmap="jet", alpha=0.7, vmin=1, vmax=4))

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")

    slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), "Slice", 0, img[0].shape[2] - 1, valinit=z, valstep=1)

    def update(val):
        z = int(slider.val)
        for i in range(4):
            imgs1[i].set_data(img[i][:, :, z])
            imgs2[i].set_data(img[i][:, :, z])
            mask1[i].set_data(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]))

            imgs3[i].set_data(img[i][:, :, z])
            mask2[i].set_data(np.ma.masked_where(prediction[:, :, z] == 0, prediction[:, :, z]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
