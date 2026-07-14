import os
import tensorflow as tf

from src.preprocessing.EDA import EDA
from src.preprocessing.loader import patients_dir
from src.preprocessing.dataset import tf_load_patient
from src.SegmentationModels.model_3UNet import Model_3UNet
from src.training.DiceLoss import Dice, Loss, callbacks
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
if not "3D_UNet.keras" in os.listdir(Data_Path + "\Models"):
    model = Model_3UNet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=Loss, metrics=[Dice])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, callbacks=callbacks)

    plot_loss(history)
    model.evaluate(test_dataset)
    model.save(Data_Path + r"Model\3D_UNet.keras")

# Results
model = tf.keras.models.load_model(Data_Path + r"\Models\3D_UNet.keras", custom_objects={"Loss": Loss, "Dice": Dice})
MRI_Results(test_dir, model)
