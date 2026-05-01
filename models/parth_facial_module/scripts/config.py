import os

# Base Path Resolution
# This script is located in parth_facial_module/scripts/
# We want the base path to be parth_facial_module/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Dataset Paths
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "processed_features.csv")
DATASET_LABELED = os.path.join(BASE_DIR, "dataset", "processed_features_labeled.csv")

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "facial_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Feature and Label Columns
FEATURE_COLUMNS = [
    'blink_rate', 
    'eye_contact_percentage', 
    'head_movement_frequency', 
    'emotion_stability', 
    'emotion_confidence'
]
LABEL_COLUMN = "confidence_label"

# Feature Ranges (Synthetic typical min/max)
FEATURE_RANGES = {
    'blink_rate': (0, 50),
    'eye_contact_percentage': (0, 100),
    'head_movement_frequency': (0, 45),
    'emotion_stability': (0.0, 1.0),
    'emotion_confidence': (0.0, 1.0)
}

# Default Values for features (sensible starting points)
DEFAULT_VALUES = {
    'blink_rate': 18,
    'eye_contact_percentage': 75,
    'head_movement_frequency': 15,
    'emotion_stability': 0.75,
    'emotion_confidence': 0.80
}

# Reproducibility
RANDOM_STATE = 42

# Confidence Classification Logic and Thresholds
# HIGH: Satisfies all conditions (High eye contact/stability, Low movement)
# LOW: Satisfies any condition (Very low eye contact/stability, High movement)
# MEDIUM: Values falling between HIGH and LOW criteria
THRESHOLDS = {
    "HIGH": {
        "eye_contact_min": 70,
        "emotion_stability_min": 0.7,
        "head_movement_max": 20
    },
    "LOW": {
        "eye_contact_max": 40,
        "emotion_stability_max": 0.4,
        "head_movement_min": 40
    }
}

# Class Names and Mappings
CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Score Mapping (Base score for each class)
SCORE_MAPPING = {"Low": 30, "Medium": 60, "High": 85}

# Validation Constants
REQUIRED_CLASSES = ["Low", "Medium", "High"]
MIN_ACCURACY_THRESHOLD = 0.6
