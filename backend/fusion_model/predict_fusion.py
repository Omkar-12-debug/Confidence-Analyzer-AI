import joblib
import os
import sys

# Add backend directory to module search path so we can import modules properly
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config

_FUSION_MODEL_CACHE = None

def load_fusion_model():
    global _FUSION_MODEL_CACHE
    if _FUSION_MODEL_CACHE is not None:
        return _FUSION_MODEL_CACHE
        
    if not config.FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Fusion model not found at {config.FUSION_MODEL_PATH}. Train it first.")
        
    _FUSION_MODEL_CACHE = joblib.load(config.FUSION_MODEL_PATH)
    return _FUSION_MODEL_CACHE

def predict_fusion(voice_score, facial_score):
    """
    Given the computed scores from the unimodal modules, predicts the final confidence class.
    """
    model = load_fusion_model()
    
    # Needs to be a 2D array: shape (1, 2)
    feature_vector = [[voice_score, facial_score]]
    
    prediction = model.predict(feature_vector)[0]
    
    return {
        "final_confidence_class": prediction,
        "features_used": {
            "voice_confidence_score": voice_score,
            "facial_confidence_score": facial_score
        }
    }

if __name__ == "__main__":
    print(predict_fusion(85.0, 90.0))
