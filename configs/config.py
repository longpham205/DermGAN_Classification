# configs/config.py
"""
Global constants and default paths for DermGAN_Classification.

NOTE:
- This file defines GLOBAL, FIXED conventions (paths, class semantics).
- Experiment-specific settings MUST come from experiment.yaml / classifier.yaml.
"""

from pathlib import Path

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================
# DATA ROOTS
# =========================
DATA_ROOT = PROJECT_ROOT / "data"            # raw datasets (ISIC, HAM10000)
DATASET_ROOT = PROJECT_ROOT / "dataset"      # processed images + splits
CONFIG_ROOT = PROJECT_ROOT / "configs"

# CONFIGS DIR
EXPERIMENT_CFG_PATH = CONFIG_ROOT / "experiment.yaml"
CLASSIFIER_CFG_PATH = CONFIG_ROOT / "classifier.yaml"
LOGGING_CFG_PATH    = CONFIG_ROOT / "logging.yaml"
GAN_CFG_PATH        = CONFIG_ROOT / "gan.yaml"

# =========================
# OUTPUT ROOTS
# =========================
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
LOG_ROOT = RESULTS_ROOT / "logs"

CLASSIFIER_CHECKPOINT_DIR = CHECKPOINT_ROOT / "classifier"

# =========================
# DEFAULTS (FALLBACK ONLY)
# =========================
DEFAULT_SEED = 42
DEFAULT_IMAGE_EXT = ".jpg"

# =========================
# CLASSES (HAM10000 / ISIC - FIXED SEMANTICS)
# =========================
CLASSES = [
    "nv",  # Actinic keratoses
    "mel",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "bcc",     # Dermatofibroma
    "akiec",    # Melanoma
    "vasc",     # Melanocytic nevi
    "df",   # Vascular lesions
]

LABEL_ORDER_SOURCE = "HAM10000 official dx taxonomy"

NUM_CLASSES = len(CLASSES)

# =========================
# LABEL MAPPING (REFERENCE)
# =========================
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# =========================
# SANITY CHECK
# =========================
def validate_num_classes(num_classes: int):
    """
    Ensure dataset / model / experiment configs are consistent.
    """
    assert num_classes == NUM_CLASSES, (
        f"NUM_CLASSES mismatch: expected {NUM_CLASSES}, got {num_classes}"
    )
