import os
from pathlib import Path
import random
import zipfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from ipywidgets import interact, IntSlider

from src.data.EDA import EDA
from src.data.loader import patients_dir
from src.data.dataset import tf_load_patient
from src.models.model_3UNet import Model_3UNet
from src.training.DiceLoss import Dice, Loss, callbacks
from src.training.mri_results import MRI_Results
from src.training.plots import plot_loss

DATA_DIR = "C:/Users/dell/Documents/Programming Projects/Python/3D Brain tumor Segmentation/Data"
patients_dir = patients_dir(DATA_DIR)

'''

if not os.path.exists(Data_Path):
    os.makedirs(Data_Path)
    with zipfile.ZipFile(Source_File, 'r') as zip_ref:
        zip_ref.extractall()


patients_dir = patients_dir(Data_Path)
random.seed(42)
random.shuffle(patients_dir)

EDA(patients_dir)

train_dir = patients_dir[:int(len(patients_dir)*0.8)]
val_dir = patients_dir[int(len(patients_dir)*0.8):int(len(patients_dir)*0.9)]
test_dir = patients_dir[int(len(patients_dir)*0.9):]


train_dataset = (tf.data.Dataset.from_tensor_slices(train_dir).map(tf_load_patient, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(1).prefetch(tf.data.AUTOTUNE))

val_dataset = (tf.data.Dataset.from_tensor_slices(val_dir).map(tf_load_patient, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(1).prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices(test_dir).map(tf_load_patient)
                  .batch(1))


model = Model_3UNet()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=Loss, metrics=[Dice])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, callbacks=callbacks)

plot_loss(history)
model.evaluate(test_dataset)

MRI_Results(patients_dir, model)

'''
