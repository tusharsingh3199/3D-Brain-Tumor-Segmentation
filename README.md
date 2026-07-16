# 3D Brain Tumor Segmentation (BraTS) — TensorFlow

End-to-end pipeline for 3D brain tumor segmentation on the BraTS 2019 dataset, progressing from a 3D U-Net baseline to a Swin-UNETR transformer.

## Project Structure

```
brain_tumor_segmentation/
├── configs/                                
│   └── config.py                           #   All parameters for the project.
│
├── src/
│   ├── data/                   
│   │   ├── loader.py                       #   Download data and gives patient directory list.
│   │   ├── EDA.py                          #   Brain tumor analysis.
│   │   └── dataset.py                      #   Preparing dataset for the models.
│   │
│   ├── models/                 
│   │   ├── model_3D_UNet.py                #   3D U-Net Model.
│   │   └── model_SwinUNETR.py              #   Swin UNETR Model.
│   │
│   ├── training/                           
│   │   ├── DiceLoss.py                     #   Class-wise Dice Score, and Loss function.
│   │   ├── mri_results.py                  #   Real and Predicted results from the Model.
│   │   └── plots.py                        #   Model training plots.
│   │
│   └── deployment/               
│       ├── model_manager.py      
│       └── api.py                
│
├── scripts/                       
│   ├── train.py                            #   Model training script.
│   ├── api.py                              #   FastAPI setup.
│   └── streamlit_app.py                    #   Streamlit UI.
│
├── Data/                       
│   ├── archive.zip                         #   Zip MRI dataset.
│   └── MICCAI_BraTS_2019_Data_Training     #   Patients file with t1, t2, t1ce, flair, seg files.
│       ├── HGG                             
│       ├── LGG
│
├── Experiment Notebooks/                       
│   └─ Segmentation_Models.ipyng           #   Notebook for experiments.
│
├── Models/                       
│   ├── 3D_UNet.keras                       #   3D UNet model weight file.
│   └── Swin_UNETR.keras                    #   Swin UNETR model weight file. 
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
