import os
import tensorflow as tf

from src.SegmentationModels import model_3D_UNet, model_SwinUNETR
from src.preprocessing.EDA import EDA
from src.preprocessing.loader import patients_dir
from src.preprocessing.dataset import tf_load_patient

from src.training.DiceLoss import *
from src.training.mri_results import MRI_Results
from src.training.plots import plot_loss
from configs.config import *

# Data Preprocess
patients_dir = patients_dir(DATA_PATH + r"\Data", seed=69)

# Exploratory Data Analysis
if Show_EDA:
    EDA(patients_dir)

# Data Preperation
train_dir = patients_dir[:int(len(patients_dir)*0.8)]
val_dir = patients_dir[int(len(patients_dir)*0.8):int(len(patients_dir)*0.9)]
test_dir = patients_dir[int(len(patients_dir)*0.9):]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_dir).map(tf_load_patient, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(1).prefetch(tf.data.AUTOTUNE))
val_dataset = (tf.data.Dataset.from_tensor_slices(val_dir).map(tf_load_patient, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(1).prefetch(tf.data.AUTOTUNE))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_dir).map(tf_load_patient)
                  .batch(1))


if "unet" in TRAIN_MODEL or (not ("3D_UNet.keras" in os.listdir(DATA_PATH + r"\Models")) and "unet" in MODELS):
    UNET_Model = model_3D_UNet.Model_3UNet()
    UNET_Model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=Loss, metrics=[Dice, Dice_NCR, Dice_ED, Dice_ET])
    history = UNET_Model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=model_3D_UNet.callbacks)

    plot_loss(history)
    UNET_Model.evaluate(test_dataset)
    UNET_Model.save(DATA_PATH + r"\Model\3D_UNet.keras")


if "swin_unetr" in TRAIN_MODEL or (not ("Swin_UNETR.keras" in os.listdir(DATA_PATH + r"\Models")) and "swin_unetr" in MODELS):
    Swin_UNETR = model_SwinUNETR.Model_SwinUNETR()
    Swin_UNETR.compile(optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-5, clipnorm=1.0), loss=Loss,
                       metrics=[Dice, Dice_NCR, Dice_ED, Dice_ET])
    history = Swin_UNETR.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=model_SwinUNETR.callbacks)

    plot_loss(history)
    Swin_UNETR.evaluate(test_dataset)
    Swin_UNETR.save(DATA_PATH + r"\Model\Swin_UNETR.keras")


# Model loading and Results
if Show_MRI:
    if "unet" in MODELS:
        UNET_Model = tf.keras.models.load_model(DATA_PATH + r"\Models\3D_UNet.keras",
                custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET})

        MRI_Results(test_dir, UNET_Model)

    if "swin_unetr" in MODELS:
        Swin_UNETR = tf.keras.models.load_model(DATA_PATH + r"\Models\Swin_UNETR.keras",
                custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET})

        MRI_Results(test_dir, Swin_UNETR)
