"""Central configuration for the toolpath signature pipeline."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "toolpath_signature_20260407"
PAPER_DIR = PROJECT_ROOT / "outputs" / "toolpath_signature_20260407" / "paper"

INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
BIN_STATS_DIR = INTERMEDIATE_DIR / "bin_stats"
AIR_REF_DIR = INTERMEDIATE_DIR / "air_reference"
ISOLATED_DIR = INTERMEDIATE_DIR / "isolated"
RUN_VECTORS_DIR = INTERMEDIATE_DIR / "run_vectors"

RESULTS_DIR = OUTPUT_DIR / "results"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
ABLATION_DIR = RESULTS_DIR / "ablation"
GENERALIZATION_DIR = RESULTS_DIR / "generalization"
IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"
PCA_DIR = RESULTS_DIR / "pca"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ── Toolpath Types & File Prefixes ─────────────────────────────────────────
TOOLPATH_TYPES = ["adaptive", "face", "pocket"]

AIR_PREFIX = {
    "adaptive": "adaptive",
    "face": "face",
    "pocket": "pocket",
}

ACTIVE_PREFIX = {
    "adaptive": "adaptive150025",
    "face": "face150025",
    "pocket": "pocket150025",
}

# ── Column Definitions ─────────────────────────────────────────────────────
ELECTRICAL_COLS = [
    "spindle", "x_motor", "y_motor", "z_motor",
    "spindle_A", "x_motor_A", "y_motor_A", "z_motor_A",
]

ARDUINO_MODULES = [
    "frame_l2", "frame_r2", "y_bed__3", "frame_l3", "spindle2", "y_bed__4",
]

ARDUINO_CHANNELS = [
    "Ax", "Ay", "Az", "Gx", "Gy", "Gz",
    "Mx", "My", "Mz",
    "Pressure", "Temperature", "Proximity",
    "ColorR", "ColorG", "ColorB", "ColorA",
    "RMS",
]

ARDUINO_COLS = [
    f"{mod}.{ch}" for mod in ARDUINO_MODULES for ch in ARDUINO_CHANNELS
]

SENSOR_COLS = ELECTRICAL_COLS + ARDUINO_COLS  # 8 + 102 = 110

DROP_COLS = [
    "t_console", "stat", "line",
    "posx", "posy", "posz", "mpox", "mpoy", "mpoz",
    "vel", "feed", "unit", "dist", "plane", "coor", "momo",
    "raw_json",
]

METADATA_COLS = ["gcode_line", "gcode_string"]

# ── Feature Extraction ─────────────────────────────────────────────────────
BLOCK_STATS = ["mean", "rms", "std", "skewness", "kurtosis", "min", "max"]
N_BINS = 20  # temporal bins

# ── Classification ─────────────────────────────────────────────────────────
CV_FOLDS = 5
RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

SVM_PARAMS = dict(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
)

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    alpha=0.01,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_STATE,
)

# ── Isolation Methods ──────────────────────────────────────────────────────
ISOLATION_METHODS = ["raw", "subtracted", "ratio", "zscore"]

# ── Sensor Groups (for ablation) ──────────────────────────────────────────
SENSOR_GROUPS = {
    "electrical": ELECTRICAL_COLS,
    "arduino_all": ARDUINO_COLS,
}
for mod in ARDUINO_MODULES:
    SENSOR_GROUPS[mod] = [f"{mod}.{ch}" for ch in ARDUINO_CHANNELS]
