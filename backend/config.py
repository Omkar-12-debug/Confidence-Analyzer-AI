"""
Centralized configuration for the Confidence Analyzer AI backend.

Path resolution is production-safe:
  • In dev mode   → paths resolve relative to the project root (unchanged behavior).
  • In frozen mode (PyInstaller exe) →
        read-only data  : resolved from sys._MEIPASS  (bundled inside the exe)
        writable data   : resolved to %APPDATA%/ConfidenceAnalyzerAI/
"""

import os
import sys
import shutil
from pathlib import Path

# ── Scikit-Learn compatibility patch for older models ──────────────
import sklearn.tree
if not hasattr(sklearn.tree.DecisionTreeClassifier, 'monotonic_cst'):
    sklearn.tree.DecisionTreeClassifier.monotonic_cst = None


# ===================================================================
#  Helper functions
# ===================================================================

def _is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, 'frozen', False)


def _get_bundle_dir() -> Path:
    """
    Directory that contains *bundled, read-only* data (models, datasets).

    Dev   → project root  (parent of backend/)
    Frozen → sys._MEIPASS  (PyInstaller extraction dir)
    """
    if _is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _get_appdata_dir() -> Path:
    """
    Writable application-data directory under the user's profile.

    Windows : %APPDATA%/ConfidenceAnalyzerAI
    Other   : ~/.local/share/ConfidenceAnalyzerAI
    """
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA',
                                   Path.home() / 'AppData' / 'Roaming'))
    else:
        base = Path.home() / '.local' / 'share'
    return base / 'ConfidenceAnalyzerAI'


def _get_writable_base() -> Path:
    """
    Base directory for *writable* artefacts (outputs, uploads, history).

    Dev    → backend/           (preserves current behaviour)
    Frozen → %APPDATA%/ConfidenceAnalyzerAI/
    """
    if _is_frozen():
        return _get_appdata_dir()
    return Path(__file__).resolve().parent          # backend/


def _get_ffmpeg_path() -> str:
    """
    Resolve the FFmpeg executable.

    Search order (frozen mode):
      1. Next to the running .exe  (Tauri bundles resources here)
      2. One level above the .exe  (alternative Tauri layout)
      3. System PATH

    Dev mode: system PATH.
    """
    if _is_frozen():
        exe_dir = Path(sys.executable).parent
        candidates = [
            exe_dir / 'ffmpeg.exe',
            exe_dir.parent / 'ffmpeg.exe',
            exe_dir.parent / 'resources' / 'ffmpeg.exe',
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())

    found = shutil.which('ffmpeg')
    return found if found else 'ffmpeg'


# ===================================================================
#  Core directories
# ===================================================================

BUNDLE_DIR     = _get_bundle_dir()        # read-only bundled data
WRITABLE_BASE  = _get_writable_base()     # writable user data

# Legacy aliases used by existing imports
BASE_DIR    = BUNDLE_DIR
BACKEND_DIR = (WRITABLE_BASE
               if _is_frozen()
               else Path(__file__).resolve().parent)


# ===================================================================
#  Read-only paths  (bundled with the app)
# ===================================================================

# Data CSVs
RAW_AUDIO_DATASET  = BUNDLE_DIR / "data" / "audio_dataset_refined.csv"
RAW_FACIAL_DATASET = BUNDLE_DIR / "data" / "processed_features.csv"

# Pre-trained model weights
AUDIO_MODEL_PATH   = BUNDLE_DIR / "models" / "heet_audio_module"  / "models" / "voice_model.pkl"
FACIAL_MODEL_PATH  = BUNDLE_DIR / "models" / "parth_facial_module" / "models" / "facial_model.pkl"
FACIAL_SCALER_PATH = BUNDLE_DIR / "models" / "parth_facial_module" / "models" / "scaler.pkl"

# Audio feature-extraction module (dynamically loaded via sys.path)
AUDIO_FEATURE_MODULE = BUNDLE_DIR / "models" / "heet_audio_module" / "audio"

# Visual module path
VISUAL_MODULE_DIR = BUNDLE_DIR / "models" / "bhavesh_visual_module"


# ===================================================================
#  Writable paths  (AppData in production, backend/ in dev)
# ===================================================================

# Output directories
AUDIO_OUTPUT_DIR  = WRITABLE_BASE / "outputs" / "audio"
FACIAL_OUTPUT_DIR = WRITABLE_BASE / "outputs" / "facial"
FUSION_OUTPUT_DIR = WRITABLE_BASE / "outputs" / "fusion"
REPORTS_DIR       = WRITABLE_BASE / "outputs" / "reports"

AUDIO_BATCH_CSV  = AUDIO_OUTPUT_DIR  / "batch_audio.csv"
FACIAL_BATCH_CSV = FACIAL_OUTPUT_DIR / "batch_facial.csv"

# Processed data
PROCESSED_DATA_DIR = WRITABLE_BASE / "data" / "processed"
FUSION_DATASET     = PROCESSED_DATA_DIR / "fusion_dataset.csv"

# Fusion model artefacts (writable so re-training is possible)
FUSION_MODEL_DIR       = WRITABLE_BASE / "fusion_model" / "artifacts"
FUSION_MODEL_PATH      = FUSION_MODEL_DIR / "fusion_model.pkl"
FUSION_METRICS_PATH    = FUSION_MODEL_DIR / "metrics.json"
FUSION_CONFUSION_MATRIX = FUSION_MODEL_DIR / "confusion_matrix.csv"

# Uploads & history
UPLOADS_DIR = WRITABLE_BASE / "uploads"
HISTORY_DIR = WRITABLE_BASE / "data" / "history"

# FFmpeg executable path
FFMPEG_PATH = _get_ffmpeg_path()


# ===================================================================
#  Create required writable directories
# ===================================================================

for _d in [AUDIO_OUTPUT_DIR, FACIAL_OUTPUT_DIR, FUSION_OUTPUT_DIR,
           REPORTS_DIR, PROCESSED_DATA_DIR, FUSION_MODEL_DIR,
           UPLOADS_DIR, HISTORY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ===================================================================
#  Seed fusion model from bundle on first run (frozen only)
# ===================================================================

if _is_frozen():
    _BUNDLED_FUSION = BUNDLE_DIR / "fusion_model_artifacts"
    if _BUNDLED_FUSION.exists() and not FUSION_MODEL_PATH.exists():
        for _f in _BUNDLED_FUSION.iterdir():
            _dest = FUSION_MODEL_DIR / _f.name
            if not _dest.exists():
                shutil.copy2(str(_f), str(_dest))
