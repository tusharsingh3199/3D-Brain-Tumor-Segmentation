import os
import tensorflow as tf

from src.SegmentationModels import model_3D_UNet, model_SwinUNETR
from src.preprocessing.EDA import EDA
from src.preprocessing.loader import patients_dir
from src.preprocessing.dataset import tf_load_patient
from src.training.DiceLoss import *
from src.training.mri_results import MRI_Results
from src.training.plots import plot_loss

# Data Preprocess
Data_Path = r"C:\Users\dell\Documents\Programming Projects\Python\3D Brain tumor Segmentation"
patients_dir = patients_dir(Data_Path + "\Data", seed=69)

# Exploratory Data Analysis
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

# Model Training
models = ["unet", "swin_unetr"]


if not ("3D_UNet.keras" in os.listdir(Data_Path + "\Models")) and "unet" in models:
    UNET_Model = model_3D_UNet.Model_3UNet()
    UNET_Model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=Loss, metrics=[Dice, Dice_NCR, Dice_ED, Dice_ET])
    history = UNET_Model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=model_3D_UNet.callbacks)

    plot_loss(history)
    UNET_Model.evaluate(test_dataset)
    UNET_Model.save(Data_Path + r"Model\3D_UNet.keras")


if not ("Swin_UNETR.keras" in os.listdir(Data_Path + "\Models")) and "swin_unetr" in models:
    Swin_UNETR = model_SwinUNETR.Model_SwinUNETR()
    Swin_UNETR.compile(optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-5, clipnorm=1.0), loss=Loss,
                       metrics=[Dice, Dice_NCR, Dice_ED, Dice_ET])
    history = Swin_UNETR.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=model_SwinUNETR.callbacks)

    plot_loss(history)
    Swin_UNETR.evaluate(test_dataset)
    Swin_UNETR.save(Data_Path + r"Model\Swin_UNETR.keras")


# Results
UNET_Model = tf.keras.models.load_model(Data_Path + r"\Models\3D_UNet.keras",
            custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET})

Swin_UNETR = tf.keras.models.load_model(Data_Path + r"\Models\Swin_UNETR.keras",
            custom_objects={"Loss": Loss, "Dice": Dice, "Dice_NCR": Dice_NCR, "Dice_ED": Dice_ED, "Dice_ET": Dice_ET})

MRI_Results(test_dir, UNET_Model)
MRI_Results(test_dir, UNET_Model)
