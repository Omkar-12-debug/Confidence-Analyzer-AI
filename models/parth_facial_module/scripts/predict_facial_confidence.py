import os
import sys
import joblib
import pandas as pd

# Ensure the scripts directory is in the path for relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import config

# Global cache for model and scaler
_MODEL_CACHE = None
_SCALER_CACHE = None

def load_artifacts():
    """Loads and caches the trained model and scaler from config paths."""
    global _MODEL_CACHE, _SCALER_CACHE
    
    if _MODEL_CACHE is not None and _SCALER_CACHE is not None:
        return _MODEL_CACHE, _SCALER_CACHE
        
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}. Please train the model first.")
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {config.SCALER_PATH}. Please train the model first.")
    
    print(f"Loading artifacts from disk...")
    _MODEL_CACHE = joblib.load(config.MODEL_PATH)
    _SCALER_CACHE = joblib.load(config.SCALER_PATH)
    
    return _MODEL_CACHE, _SCALER_CACHE

def preprocess_input(input_dict):
    """Cleans, clips, and scales the input dictionary for prediction."""
    processed_features = {}
    
    # 1. Input Handling & Defaults with Warnings
    for col in config.FEATURE_COLUMNS:
        if col not in input_dict:
            print(f"Warning: Missing input feature '{col}'. Using default value: {config.DEFAULT_VALUES.get(col)}")
            val = config.DEFAULT_VALUES.get(col, 0.0)
        else:
            val = input_dict[col]
        
        # 2. Safe Float Conversion
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = float(config.DEFAULT_VALUES.get(col, 0.0))
            
        # 3. Clipping
        if col in config.FEATURE_RANGES:
            min_val, max_val = config.FEATURE_RANGES[col]
            val = max(min_val, min(max_val, val))
            
        processed_features[col] = val
        
    # 4. Convert to DataFrame (ensure correct column order)
    features_df = pd.DataFrame([processed_features])[config.FEATURE_COLUMNS]
    
    return features_df

def predict_confidence(input_features):
    """Main function to predict facial confidence and return formatted output."""
    try:
        # Load artifacts (uses cache)
        model, scaler = load_artifacts()
        
        # Preprocess
        features_df = preprocess_input(input_features)
        
        # Transform using pre-fitted scaler
        features_scaled = scaler.transform(features_df)
        
        # Predict numeric class
        prediction = model.predict(features_scaled)[0]
        
        # 5. Strict Prediction Index Validation
        if int(prediction) not in config.REVERSE_CLASS_MAPPING:
            raise ValueError(f"Invalid predicted class index: {prediction}")
            
        # Map back to class name
        class_name = config.REVERSE_CLASS_MAPPING.get(int(prediction), "Unknown")
        
        # Map class name to numeric score
        score = config.SCORE_MAPPING.get(class_name, 0)
        
        return {
            "facial_confidence_score": int(score),
            "confidence_class": str(class_name)
        }
        
    except Exception as e:
        raise RuntimeError(f"Facial confidence prediction failed: {e}")

if __name__ == "__main__":
    # Test caching and warnings
    sample_input = {
        "blink_rate": 15.0,
        "eye_contact_percentage": 85.0,
        # missing head_movement_frequency
        "emotion_stability": 0.9,
        "emotion_confidence": 0.8
    }
    
    print("Testing Facial Confidence Prediction (Call 1)...")
    try:
        print(f"Result: {predict_confidence(sample_input)}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nTesting Facial Confidence Prediction (Call 2 - should use cache)...")
    try:
        print(f"Result: {predict_confidence(sample_input)}")
    except Exception as e:
        print(f"Error: {e}")
