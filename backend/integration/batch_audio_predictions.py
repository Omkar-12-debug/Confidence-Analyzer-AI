import pandas as pd
import joblib
import os
import sys

# Add backend directory to module search path so we can import modules properly
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config

def calculate_voice_score(features, max_prob):
    # Post correction logic to fix unrealistic 100 scores
    model_score = max_prob * 100
    speech_score = min(max((features["speech_rate"] / 6) * 100, 0), 100)
    pause_score = 100 - (features["pause_ratio"] * 100)
    
    # Fix energy scaling (avoid multiplying by 1e6 which saturates to 100 immediately)
    # Average energy is ~0.002 to ~0.02. Scale to 0-100 sensibly.
    energy_val = features["energy"]
    energy_score = min((energy_val / 0.02) * 100, 100)
    
    final_score = (
        0.5 * model_score +
        0.2 * speech_score +
        0.2 * pause_score +
        0.1 * energy_score
    )
    return max(0, min(final_score, 100))

def run_batch():
    if not config.RAW_AUDIO_DATASET.exists():
        print(f"Error: dataset {config.RAW_AUDIO_DATASET} not found.")
        return
        
    print(f"Loading audio dataset: {config.RAW_AUDIO_DATASET}")
    df = pd.read_csv(config.RAW_AUDIO_DATASET)
    
    print(f"Loading audio model: {config.AUDIO_MODEL_PATH}")
    model = joblib.load(config.AUDIO_MODEL_PATH)
    
    required_features = ["pitch_mean", "pitch_std", "energy", "mfcc_mean", "pause_ratio", "speech_rate"]
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            # Prepare feature vector
            feature_vector = pd.DataFrame([row[required_features]])
            
            # Predict
            prediction = model.predict(feature_vector)[0]
            raw_probs = model.predict_proba(feature_vector)[0]
            prob_sum = sum(raw_probs)
            max_prob = max(raw_probs) / prob_sum if prob_sum > 0 else 0.0
            
            # Post correction result
            final_prediction = prediction
            if max_prob < 0.6:
                if row["pause_ratio"] > 0.5:
                    final_prediction = "nervous"
                elif row["speech_rate"] < 3.0:
                    final_prediction = "neutral"
                else:
                    final_prediction = "neutral"
            
            score = calculate_voice_score(row, max_prob)
            
            results.append({
                "source_file": row.get("source_file", f"row_{idx}"),
                "audio_prediction": final_prediction,
                "model_prediction": prediction,
                "model_confidence": round(max_prob * 100, 2),
                "voice_confidence_score": round(score, 2),
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append({
                "source_file": row.get("source_file", f"row_{idx}"),
                "audio_prediction": "error",
                "model_prediction": "error",
                "model_confidence": None,
                "voice_confidence_score": None,
            })
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(config.AUDIO_BATCH_CSV, index=False)
    print(f"Batch audio predictions saved to {config.AUDIO_BATCH_CSV}")

if __name__ == "__main__":
    run_batch()
