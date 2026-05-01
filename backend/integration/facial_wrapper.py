import sys
import os
import importlib.util
from pathlib import Path

# Add backend to path so we can import backend config correctly if needed
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

import config as backend_config

def load_isolated_facial_predictor():
    """
    Safely loads the standalone parth_facial predictions script
    without causing namespace collisions on 'config.py'.
    """
    parth_module_dir = str(backend_config.BASE_DIR / "models" / "parth_facial_module" / "scripts")
    
    # Remove existing global config to prevent the facial module from loading backend/config.py
    original_config = sys.modules.pop('config', None)
    original_predict = sys.modules.pop('predict_facial_confidence', None)
    
    # Temporarily prepend parth module scripts to path to allow absolute 'import config' inside the module
    sys.path.insert(0, parth_module_dir)
    
    import predict_facial_confidence
    
    # Remove from sys.path
    sys.path.pop(0)
    
    # Keep the imported facial module
    facial_predictor = predict_facial_confidence
    
    # Restore the original backend config
    if original_config:
        sys.modules['config'] = original_config
    else:
        sys.modules.pop('config', None)
        
    return facial_predictor

# Instantiate loaded predictor
facial_predictor = load_isolated_facial_predictor()

def predict_confidence(input_dict):
    return facial_predictor.predict_confidence(input_dict)
