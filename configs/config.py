import tensorflow as tf

# Data Paths
DATA_PATH = r"C:\Users\dell\Documents\Programming Projects\Python\3D Brain tumor Segmentation"
TUMOR = ["HGG", "LGG"]
MODALITIES = ["t1", "t1ce", "t2", "flair"]


# Data Sizes
VOLUME_SIZE = (240, 240, 155, 4)
IMAGE_SHAPE = (128, 128, 128, 4)
MASK_SHAPE = (128, 128, 128)
CLASSES = len(MODALITIES)
PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)
CLASS_WEIGHTS = tf.constant([0.05, 0.25, 0.35, 0.35], dtype=tf.float32)


# Training Parameters
MODELS = ["unet", "swin_unetr"]
Show_EDA = False
EPOCHS = 10
TRAIN_MODEL = []
Show_MRI = False


# Swin UNETR parameters
patch_size = 2
embed_dim = 48
depths = (2, 2, 2, 2)
num_heads = (3, 6, 12, 24)
window_size = (4, 4, 4)



