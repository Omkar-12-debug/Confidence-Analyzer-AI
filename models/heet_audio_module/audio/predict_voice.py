import os
import sys
import joblib
import pandas as pd
import json

# Fix pathing and imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HEET_MODULE_DIR = os.path.dirname(CURRENT_DIR)
if HEET_MODULE_DIR not in sys.path:
    sys.path.append(HEET_MODULE_DIR)

from audio.audio_module import AudioRecorder
from audio.feature_extraction import extract_all_features

MODEL_PATH = os.path.join(HEET_MODULE_DIR, "models", "voice_model.pkl")


def predict_voice_confidence():

    print("\nSpeak your answer clearly...\n")

    # Record audio
    recorder = AudioRecorder(duration=5)
    audio_signal = recorder.record_audio()

    # Extract features
    features = extract_all_features(audio_signal, recorder.sample_rate)

    print("\nExtracted Features:")
    print(features)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Prepare feature vector with feature names
    feature_vector = pd.DataFrame([{
        "pitch_mean": features["pitch_mean"],
        "pitch_std": features["pitch_std"],
        "energy": features["energy"],
        "mfcc_mean": features["mfcc_mean"],
        "pause_ratio": features["pause_ratio"],
        "speech_rate": features["speech_rate"]
    }])

    # Predict class and probability
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    max_prob = max(probabilities)

    # STEP 2 - POST-PREDICTION CORRECTION LOGIC
    if max_prob < 0.6:
        if features["pause_ratio"] > 0.5:
            final_prediction = "nervous"
        elif features["speech_rate"] < 3.0:
            final_prediction = "neutral"
        else:
            final_prediction = "neutral"
    else:
        final_prediction = prediction

    # STEP 2 — COMPUTE FINAL CONFIDENCE SCORE
    model_score = max_prob * 100
    speech_score = min(max((features["speech_rate"] / 6) * 100, 0), 100)
    pause_score = 100 - (features["pause_ratio"] * 100)
    energy_score = min(features["energy"] * 1e6, 100)

    final_score = (
        0.5 * model_score +
        0.2 * speech_score +
        0.2 * pause_score +
        0.1 * energy_score
    )

    # Clamp to valid range [0, 100]
    final_score = max(0, min(final_score, 100))

    # STEP 3 — OUTPUT FORMAT
    print("\n--- Model Output ---")
    print(f"Model Prediction: {prediction}")
    print(f"Model Confidence: {max_prob * 100:.2f} %")

    print("\n--- Post-Correction Result ---")
    print(f"Final Prediction: {final_prediction}")

    print("\n--- Final Score ---")
    print(f"Voice Confidence Score: {final_score:.2f}")

    # --- JSON OUTPUT ---
    json_output = {
        "audio_prediction": final_prediction,
        "model_prediction": prediction,
        "model_confidence": round(max_prob * 100, 2),
        "voice_confidence_score": round(final_score, 2),
        "features": features
    }
    print("\n--- JSON Result ---")
    print(json.dumps(json_output, indent=4))

    # STEP 5 — FINAL STATUS
    print("\nAUDIO MODULE COMPLETE — READY FOR MULTIMODAL INTEGRATION")



if __name__ == "__main__":
    predict_voice_confidence()