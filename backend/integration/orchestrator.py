import sys
import os
import json
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config
from integration.batch_audio_predictions import calculate_voice_score
import joblib
from fusion_model.predict_fusion import predict_fusion

# Import isolated facial predictor
from integration.facial_wrapper import predict_confidence as predict_facial_confidence

# Load audio model safely once
_AUDIO_MODEL = None
def get_audio_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        _AUDIO_MODEL = joblib.load(config.AUDIO_MODEL_PATH)
    return _AUDIO_MODEL


# ---------------------------------------------------------------------------
# Threshold-based confidence class from numeric score
# ---------------------------------------------------------------------------
def score_to_class(score):
    """Derive confidence class label strictly from the numeric score.
    Thresholds:  0–39 → Low,  40–69 → Medium,  70–100 → High
    """
    if score is None:
        return "N/A"
    score = round(float(score))
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# FFmpeg video conversion: webm → mp4 for OpenCV compatibility
# ---------------------------------------------------------------------------
def convert_video_to_mp4(input_path):
    """
    Convert a video file to H.264 mp4 using FFmpeg.
    Returns (output_path, success).  If conversion fails or the input is
    already mp4, returns the original path.
    """
    input_path = str(input_path)
    ext = os.path.splitext(input_path)[1].lower()

    # Skip conversion if already mp4
    if ext == ".mp4":
        logger.info(f"[convert] Input is already .mp4, skipping conversion: {input_path}")
        return input_path, True

    output_path = os.path.splitext(input_path)[0] + "_converted.mp4"
    logger.info(f"[convert] Converting video for OpenCV compatibility")
    logger.info(f"[convert]   source : {input_path}")
    logger.info(f"[convert]   target : {output_path}")

    # Verify ffmpeg is available
    ffmpeg_cmd = config.FFMPEG_PATH
    if not os.path.isfile(ffmpeg_cmd) and not shutil.which(ffmpeg_cmd):
        logger.error("[convert] FFmpeg not found — cannot convert video")
        return input_path, False

    cmd = [
        ffmpeg_cmd,
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",   # speed over compression
        "-c:a", "aac",
        "-y",                      # overwrite without asking
        output_path,
    ]
    logger.info(f"[convert] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,            # hard safety cap
        )
        if result.returncode != 0:
            stderr_tail = result.stderr.decode(errors="replace")[-500:]
            logger.error(f"[convert] FFmpeg failed (rc={result.returncode}): {stderr_tail}")
            return input_path, False

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("[convert] FFmpeg produced an empty or missing output file")
            return input_path, False

        logger.info(f"[convert] Conversion succeeded — output size: {os.path.getsize(output_path)} bytes")
        return output_path, True

    except subprocess.TimeoutExpired:
        logger.error("[convert] FFmpeg conversion timed out (>120s)")
        return input_path, False
    except Exception as e:
        logger.error(f"[convert] Unexpected error during conversion: {e}")
        return input_path, False


def run_pipeline(payload):
    """
    Original pipeline: accepts pre-extracted feature JSON.
    Kept for backward compatibility.
    """
    audio_features = payload.get("audio_features", {})
    facial_features = payload.get("facial_features", {})
    source_id = payload.get("source_id", "live_stream")

    audio_result = _run_audio_from_features(audio_features)
    facial_result = _run_facial_from_features(facial_features)
    fusion_result = _run_fusion(audio_result, facial_result)

    final_result = {
        "source_id": source_id,
        "timestamp": datetime.now().isoformat(),
        "audio_module": audio_result,
        "facial_module": facial_result,
        "fusion_module": fusion_result,
        "warnings": []
    }

    _save_result(final_result)
    return final_result


def run_pipeline_from_file(video_path, source_id="recording"):
    """
    Full pipeline: extracts features from an actual video file,
    runs audio model, facial model, and fusion model.
    """
    from integration.extract_audio_from_video import extract_audio_features_from_file
    from integration.extract_visual_from_video import extract_visual_features_from_file

    warnings = []
    converted_path = None  # track so we can clean up later

    # 1. Extract audio features from the ORIGINAL file (webm audio works fine)
    logger.info(f"Extracting audio features from {video_path}")
    audio_extraction = extract_audio_features_from_file(str(video_path))

    if audio_extraction["status"] == "success":
        audio_features = audio_extraction["features"]
        audio_result = _run_audio_from_features(audio_features)
        audio_result["features"] = audio_features
        audio_result["duration"] = audio_extraction.get("duration")
    elif audio_extraction["status"] == "audio_insufficient_speech":
        # Near-silent or speech-absent recording — do NOT score
        warnings.append(audio_extraction.get("error",
                        "No sufficient speech detected. Please speak clearly and try again."))
        audio_result = {
            "status": "audio_insufficient_speech",
            "score": None,
            "message": audio_extraction.get("error"),
            "details": audio_extraction.get("details", []),
            "duration": audio_extraction.get("duration"),
        }
        audio_features = audio_extraction.get("features", {})
    else:
        warnings.append(f"Audio extraction failed: {audio_extraction.get('error', 'unknown')}")
        audio_result = {"status": "error", "error": audio_extraction.get("error"), "score": 0.0}
        audio_features = {}

    # 2. Convert video to mp4 for OpenCV compatibility, then extract visual features
    visual_file = str(video_path)
    converted_path_str, conversion_ok = convert_video_to_mp4(video_path)
    if conversion_ok and converted_path_str != str(video_path):
        visual_file = converted_path_str
        converted_path = converted_path_str   # mark for cleanup
        logger.info(f"[pipeline] Using converted mp4 for visual analysis: {visual_file}")
    elif not conversion_ok:
        warnings.append("Video conversion to mp4 failed — visual analysis may fail")
        logger.warning("[pipeline] Conversion failed, falling back to original file for visual analysis")

    try:
        logger.info(f"Extracting visual features from {visual_file}")
        visual_extraction = extract_visual_features_from_file(visual_file)

        if visual_extraction["status"] == "facial_no_face_detected":
            # No face found — do NOT run facial model on fake features
            facial_result = {
                "status": "facial_no_face_detected",
                "score": None,
                "class": None,
                "features": None,
                "frames_processed": visual_extraction.get("frames_processed"),
                "face_detected_frames": 0,
                "message": visual_extraction.get("error",
                           "No face detected. Please keep your face visible and try again."),
            }
            logger.info("[pipeline] No face detected — facial scoring skipped")
            if visual_extraction.get("warnings"):
                warnings.extend(visual_extraction["warnings"])
        elif visual_extraction["status"] in ("success", "warning"):
            facial_features = visual_extraction["features"]
            facial_result = _run_facial_from_features(facial_features)
            facial_result["features"] = facial_features
            facial_result["frames_processed"] = visual_extraction.get("frames_processed")
            facial_result["face_detected_frames"] = visual_extraction.get("face_detected_frames")
            logger.info(f"[pipeline] Visual analysis complete — "
                        f"frames_processed={visual_extraction.get('frames_processed')}, "
                        f"face_detected_frames={visual_extraction.get('face_detected_frames')}")
            if visual_extraction.get("warnings"):
                warnings.extend(visual_extraction["warnings"])
        else:
            warnings.append(f"Visual extraction failed: {visual_extraction.get('error', 'unknown')}")
            facial_result = {"status": "error", "error": visual_extraction.get("error"), "score": None}
    finally:
        # Clean up the converted temp file
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                logger.info(f"[pipeline] Cleaned up converted file: {converted_path}")
            except Exception:
                pass

    # 3. Fusion — audio is MANDATORY for overall confidence
    fusion_result = _run_fusion(audio_result, facial_result)
    analysis_mode = fusion_result.get("analysis_mode", "invalid_recording")

    final_result = {
        "source_id": source_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": analysis_mode,
        "audio_module": audio_result,
        "facial_module": facial_result,
        "fusion_module": fusion_result,
        "warnings": warnings
    }

    _save_result(final_result)
    return final_result


def _run_audio_from_features(audio_features):
    """Run audio model on pre-extracted features."""
    if not audio_features:
        return {"status": "skipped", "score": None}
    try:
        model = get_audio_model()
        import pandas as pd
        required = ["pitch_mean", "pitch_std", "energy", "mfcc_mean", "pause_ratio", "speech_rate"]
        row = pd.DataFrame([{f: audio_features.get(f, 0) for f in required}])

        prediction = model.predict(row)[0]
        raw_probs = model.predict_proba(row)[0]
        prob_sum = sum(raw_probs)
        max_prob = max(raw_probs) / prob_sum if prob_sum > 0 else 0.0

        score = calculate_voice_score(audio_features, max_prob)

        return {
            "status": "success",
            "score": round(float(score), 2),
            "model_prediction": prediction,
            "model_confidence": round(float(max_prob) * 100, 2)
        }
    except Exception as e:
        logger.error(f"Audio prediction error: {e}")
        return {"status": "error", "error": str(e), "score": 0.0}


def _run_facial_from_features(facial_features):
    """Run facial model on pre-extracted features."""
    if not facial_features:
        return {"status": "skipped", "score": None}
    try:
        res = predict_facial_confidence(facial_features)
        return {
            "status": "success",
            "score": float(res.get("facial_confidence_score", 0)),
            "class": res.get("confidence_class")
        }
    except Exception as e:
        logger.error(f"Facial prediction error: {e}")
        return {"status": "error", "error": str(e), "score": 0.0}


def _run_fusion(audio_result, facial_result):
    """Run fusion logic with audio as MANDATORY for overall confidence.

    Analysis modes:
      full_multimodal        – audio valid + facial valid  → ML fusion
      voice_only             – audio valid + facial invalid → voice score only
      facial_only_incomplete – audio invalid + facial valid → NO overall score
      invalid_recording      – both invalid                → NO overall score

    The final displayed label is ALWAYS derived from the numeric score
    using fixed thresholds (0-39=Low, 40-69=Medium, 70-100=High).
    The ML fusion model's raw class prediction is kept as debug info only.
    """
    v_score = audio_result.get("score")
    f_score = facial_result.get("score")

    audio_status = audio_result.get("status", "")
    facial_status = facial_result.get("status", "")

    # Audio is considered valid only if status is 'success' and score exists
    has_audio = (audio_status == "success" and v_score is not None)
    # Facial is valid if status is 'success' and score exists
    has_facial = (facial_status == "success" and f_score is not None)

    # -----------------------------------------------------------------
    # CASE 1: Full multimodal — both modalities valid
    # -----------------------------------------------------------------
    if has_audio and has_facial:
        final_score = round((float(v_score) + float(f_score)) / 2, 2)
        final_class = score_to_class(final_score)

        try:
            fusion_output = predict_fusion(float(v_score), float(f_score))
            fusion_output["fusion_model_raw_class"] = fusion_output.pop("final_confidence_class", None)
            fusion_output["final_confidence_score"] = final_score
            fusion_output["final_confidence_class"] = final_class
            fusion_output["status"] = "success"
            fusion_output["analysis_mode"] = "full_multimodal"
            fusion_output["message"] = "Full Multimodal Analysis"
            return fusion_output
        except Exception as e:
            logger.error(f"Fusion ML model error: {e}")
            return {
                "status": "success",
                "analysis_mode": "full_multimodal",
                "message": "Full Multimodal Analysis",
                "final_confidence_score": final_score,
                "final_confidence_class": final_class,
                "features_used": {
                    "voice_confidence_score": v_score,
                    "facial_confidence_score": f_score
                }
            }

    # -----------------------------------------------------------------
    # CASE 2: Voice-only — audio valid, facial missing/invalid
    # -----------------------------------------------------------------
    if has_audio and not has_facial:
        final_score = round(float(v_score), 2)
        final_class = score_to_class(final_score)
        return {
            "status": "voice_only",
            "analysis_mode": "voice_only",
            "message": "No face detected. Result is based on voice only.",
            "final_confidence_score": final_score,
            "final_confidence_class": final_class,
            "features_used": {
                "voice_confidence_score": v_score,
                "facial_confidence_score": None
            }
        }

    # -----------------------------------------------------------------
    # CASE 3: Facial-only incomplete — audio invalid, facial valid
    # -----------------------------------------------------------------
    if not has_audio and has_facial:
        return {
            "status": "incomplete",
            "analysis_mode": "facial_only_incomplete",
            "message": "Speech is required for confidence evaluation. Facial cues alone are not enough.",
            "final_confidence_score": None,
            "final_confidence_class": "Incomplete",
            "features_used": {
                "voice_confidence_score": None,
                "facial_confidence_score": f_score
            }
        }

    # -----------------------------------------------------------------
    # CASE 4: Invalid recording — both modalities invalid
    # -----------------------------------------------------------------
    return {
        "status": "incomplete",
        "analysis_mode": "invalid_recording",
        "message": "No sufficient speech or face detected. Please record again.",
        "final_confidence_score": None,
        "final_confidence_class": "Incomplete",
        "features_used": {
            "voice_confidence_score": None,
            "facial_confidence_score": None
        }
    }


def _save_result(result):
    """Save result as latest and append to history."""
    try:
        # Save latest
        with open(config.REPORTS_DIR / "latest_result.json", "w") as f:
            json.dump(result, f, indent=4, default=str)

        # Append to history
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = config.HISTORY_DIR / f"result_{ts}.json"
        with open(history_file, "w") as f:
            json.dump(result, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
