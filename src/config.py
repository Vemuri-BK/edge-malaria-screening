"""
Central configuration for the EdAi malaria edge project.
Every other module imports its settings from here, so experiments
are controlled from one place.
"""
import os
from pathlib import Path

# ---- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]      # project root (EdAi_project/)
SPLITS_DIR = ROOT / "splits"                     # split CSVs (committed to git)
OUTPUT_DIR = ROOT / "outputs"                    # models, logs, plots (not committed)

# Default dataset location on Kaggle (folder that contains Parasitized/ and Uninfected/)
KAGGLE_DATA_ROOT = "/kaggle/input/cell-images-for-detecting-malaria/cell_images"

# ---- Data --------------------------------------------------------------
# Index 0 = negative, index 1 = positive (what the red LED means)
CLASSES = ["Uninfected", "Parasitized"]
IMG_SIZE = 128                       # cells are small (~100-150 px); 128 keeps the Pi fast
SPLIT_RATIOS = (0.70, 0.15, 0.15)    # train / val / test

# ---- Reproducibility ---------------------------------------------------
SEED = 42

# ---- Model + training (Step 3-4) ---------------------------------------
ARCH = "mobilenet_v3_small"          # also: "mobilenet_v2", "resnet18"
PRETRAINED = True                    # ImageNet weights (needs internet on Kaggle)
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = min(4, os.cpu_count() or 2)

# ---- LED decision rule (Step 6) ----------------------------------------
# The LED turns ON when P(Parasitized) >= threshold.
# The threshold is chosen on the VALIDATION set as the highest value that
# still catches at least TARGET_SENSITIVITY of infected cells
# (missing an infection is worse than a false alarm).
TARGET_SENSITIVITY = 0.98

# ---- Quantization (Step 8) ---------------------------------------------
CALIB_IMAGES = 300                   # training images used to calibrate INT8 ranges
