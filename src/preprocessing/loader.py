import os
import zipfile
import kaggle
import random
from configs.config import *


def download_data(data_dir):
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print("Data already exists...")
        return
    os.makedirs(data_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("aryashah2k/brain-tumor-segmentation-brats-2019", path=data_dir)


def patients_dir(Data_Path, seed=42):
    download_data(Data_Path)

    if len(os.listdir(Data_Path)) < 2:
        Source_File = os.path.join(Data_Path, os.listdir(Data_Path)[0])
        print("Extracting Data....")
        with zipfile.ZipFile(Source_File, 'r') as zip_ref:
            zip_ref.extractall(Data_Path)

    folders = [f for f in os.listdir(Data_Path) if os.path.isdir(os.path.join(Data_Path, f))]
    Data_Path = os.path.join(Data_Path, folders[0])

    patient_dir = []
    for grade in TUMOR:
        grade_dir = os.path.join(Data_Path, grade)
        patients = sorted(os.listdir(grade_dir))
        patient_paths = [os.path.join(grade_dir, p) for p in patients]

        for path in patient_paths:
            patient_dir.append(path)

    sorted(patient_dir)
    random.seed(seed)
    random.shuffle(patient_dir)

    return patient_dir

