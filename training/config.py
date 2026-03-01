"""Shared configuration for ArduRevo training pipeline."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ArduRevo fotky usporiadane" / "fotky"
MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# --- Classes ---
CLASS_NAMES = ["dolava", "rovno", "doprava"]  # folder names = labels
NUM_CLASSES = 3

# Map for horizontal flip label swap: left <-> right, straight stays
FLIP_LABEL_MAP = {0: 2, 1: 1, 2: 0}  # dolava <-> doprava

# --- Model input ---
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_CHANNELS = 1  # grayscale

# --- Preprocessing ---
CROP_BOTTOM_PCT = 0.15  # crop bottom 15% of image (rover hardware visible)

# --- Training ---
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
EARLY_STOPPING_PATIENCE = 25
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5

# --- Augmentation ---
ROTATION_RANGE = 10       # degrees
BRIGHTNESS_RANGE = 0.3    # +/- 30%
CONTRAST_RANGE = 0.2      # +/- 20%
RANDOM_CROP_PCT = 0.10    # crop up to 10%
GAUSSIAN_NOISE_STD = 0.02 # normalized pixel space

# --- Export ---
TFLITE_MODEL_PATH = OUTPUT_DIR / "model.tflite"
C_HEADER_PATH = PROJECT_ROOT / "esp32-cam" / "include" / "model_data.h"
