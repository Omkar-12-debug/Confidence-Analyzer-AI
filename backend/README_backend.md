# Confidence Analyzer AI Backend

This is the fully integrated backend pipeline for the multimodal Confidence Analyzer AI project. It includes batch inference scripts, the final fusion model pipeline, and a live orchestrator mapped to a Flask API.

## Directory Structure

- `backend/api/` - Flask API application
- `backend/config.py` - Global paths and configuration
- `backend/data/` - Holds processed datasets
- `backend/fusion_model/` - Scripts to train and run the fusion model
- `backend/integration/` - Core logic mapping unimodal outputs to the multimodal pipeline
- `backend/outputs/` - Generated batch files, reports, and ML artifacts

## Running Batch Processing

To rebuild the fusion dataset from scratch:
1. `python backend/integration/batch_audio_predictions.py`
2. `python backend/integration/batch_facial_predictions.py`
3. `python backend/integration/build_fusion_dataset.py`
4. `python backend/fusion_model/train_fusion_model.py`

## Running Live System

Start the Flask backend:
```bash
python run_backend.py
```

## API Documentation

- `GET /api/health` - Check if server is running
- `POST /api/analyze` - Run multimodal pipeline on a set of features.
Payload format:
```json
{
  "source_id": "optional_string",
  "audio_features": {
    "pitch_mean": 120.5,
    "pitch_std": 30.1,
    "energy": 0.002,
    "mfcc_mean": -6.5,
    "pause_ratio": 0.2,
    "speech_rate": 5.5
  },
  "facial_features": {
    "blink_rate": 18,
    "eye_contact_percentage": 85,
    "head_movement_frequency": 15,
    "emotion_stability": 0.9,
    "emotion_confidence": 0.8
  }
}
```
- `GET /api/latest-result` - Fetches the last completed prediction.
- `POST /api/train-fusion` - Retrains the fusion model.
