# config.py

import os

# ----------------------
# Project Paths
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model saving path
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if not exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)


# ----------------------
# Training Parameters
# ----------------------
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 50

# Input shape (3D MRI patches, e.g., 128x128x128)
INPUT_SHAPE = (128, 128, 128)

# Number of channels: MRI scans are often 1 channel (grayscale),
# but some datasets have multiple modalities (T1, T2, FLAIR).
N_CHANNELS = 1

# Binary segmentation: Tumor vs. Non-tumor
N_CLASSES = 2


# ----------------------
# Data Parameters
# ----------------------
# Train/Validation split
VAL_SPLIT = 0.2

# Random seed for reproducibility
SEED = 42
