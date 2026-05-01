"""
Extract audio features from an uploaded video/audio file.
Reuses the feature extraction logic from heet_audio_module.
"""
import sys
import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)
import config

# Add the audio module to path so we can reuse its feature extraction
audio_module_path = str(config.AUDIO_FEATURE_MODULE)
if audio_module_path not in sys.path:
    sys.path.insert(0, audio_module_path)


def extract_audio_features_from_file(video_path):
    """
    Extract audio features from a video or audio file.
    Returns dict with status, features, and duration.
    
    Strategy:
    1. Try librosa.load() directly (works if audioread/ffmpeg backend available)
    2. If that fails, try ffmpeg subprocess to extract wav first
    3. Then compute features using the same logic as heet_audio_module
    """
    import numpy as np
    
    audio_signal = None
    sr = 22050
    temp_wav = None

    # Strategy 1: Try librosa directly
    try:
        import librosa
        audio_signal, sr = librosa.load(str(video_path), sr=22050, mono=True)
        logger.info(f"Loaded audio directly via librosa: {len(audio_signal)} samples")
    except Exception as e:
        logger.warning(f"Direct librosa load failed: {e}. Trying ffmpeg extraction...")

    # Strategy 2: ffmpeg extraction fallback
    if audio_signal is None or len(audio_signal) == 0:
        try:
            temp_wav = os.path.join(str(config.UPLOADS_DIR), "temp_audio.wav")
            result = subprocess.run(
                [config.FFMPEG_PATH, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
                 '-ar', '22050', '-ac', '1', temp_wav, '-y'],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace')
                logger.error(f"ffmpeg failed: {stderr}")
                return {"status": "error", "error": f"ffmpeg extraction failed: {stderr[:200]}"}
            
            import librosa
            audio_signal, sr = librosa.load(temp_wav, sr=22050, mono=True)
            logger.info(f"Loaded audio via ffmpeg extraction: {len(audio_signal)} samples")
        except FileNotFoundError:
            return {"status": "error", "error": "Neither librosa nor ffmpeg could extract audio. Install ffmpeg."}
        except Exception as e:
            return {"status": "error", "error": f"Audio extraction failed: {str(e)}"}
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

    if audio_signal is None or len(audio_signal) == 0:
        return {"status": "error", "error": "Empty audio signal extracted"}

    # Compute features using the same functions as heet_audio_module
    try:
        from feature_extraction import extract_all_features
        features = extract_all_features(audio_signal.astype(np.float32), sr)
        logger.info("[audio] Successfully imported and used real heet feature_extraction.")
    except ImportError as e:
        logger.warning(f"[audio] Could not import heet feature_extraction ({e}), using inline fallback implementation.")
        features = _extract_features_inline(audio_signal.astype(np.float32), sr)

    duration = len(audio_signal) / sr

    # ── Speech presence validation ─────────────────────────────────────
    # Reject near-silent or speech-absent recordings BEFORE scoring.
    speech_check = _validate_speech_presence(audio_signal, sr, features, duration)
    if not speech_check["has_speech"]:
        logger.warning(f"Insufficient speech detected: {speech_check['reasons']}")
        return {
            "status": "audio_insufficient_speech",
            "error": "No sufficient speech detected. Please speak clearly and try again.",
            "details": speech_check["reasons"],
            "features": features,
            "duration": round(duration, 2),
        }

    return {
        "status": "success",
        "features": features,
        "duration": round(duration, 2)
    }


def _validate_speech_presence(audio_signal, sr, features, duration_seconds):
    """
    Determine whether the audio contains sufficient speech to score.

    Uses a weighted/combined approach to avoid brittle rejection:
      - If speech rate and pause ratio strongly indicate speech, allow a borderline RMS miss.
      - Base RMS_THRESHOLD lowered to 0.035.
    """
    import numpy as np

    # ── Thresholds ──────────────────────────────────────────────────
    BASE_RMS_THRESHOLD    = 0.035   # lowered from 0.05
    RELAXED_RMS_THRESHOLD = 0.025   # allowed if other indicators are strong
    MIN_SPEECH_RATE       = 1.0     # onsets/s; conversational speech ≈ 3–6
    MAX_PAUSE_RATIO       = 0.60    # at most 60% silent frames allowed

    reasons = []
    checks = {}   # per-check pass/fail for debug logging

    rms = float(np.sqrt(np.mean(audio_signal ** 2)))
    pause_ratio = features.get("pause_ratio", 1.0)
    speech_rate = features.get("speech_rate", 0.0)

    # ── Check Strong Indicators ─────────────────────────────────────
    # If pause ratio is very low (e.g. <= 0.45) and speech rate is good (e.g. >= 2.0)
    strong_speech_indicators = (pause_ratio <= 0.45) and (speech_rate >= 2.0)

    # Determine applicable RMS threshold
    rms_threshold_to_use = RELAXED_RMS_THRESHOLD if strong_speech_indicators else BASE_RMS_THRESHOLD

    # ── Validations ───────────────────────────────────────────────
    rms_ok = rms >= rms_threshold_to_use
    checks["rms"] = rms_ok
    if not rms_ok:
        reasons.append(f"Audio RMS energy too low ({rms:.5f} < {rms_threshold_to_use})")

    pause_ok = pause_ratio <= MAX_PAUSE_RATIO
    checks["pause_ratio"] = pause_ok
    if not pause_ok:
        reasons.append(f"Pause ratio too high ({pause_ratio:.3f} > {MAX_PAUSE_RATIO})")

    rate_ok = speech_rate >= MIN_SPEECH_RATE
    checks["speech_rate"] = rate_ok
    if not rate_ok:
        reasons.append(f"Speech rate too low ({speech_rate:.2f} < {MIN_SPEECH_RATE} onsets/s)")

    # ── Verdict: ALL conditions must pass ───────────────────────────
    has_speech = rms_ok and pause_ok and rate_ok

    # ── Debug logging ───────────────────────────────────────────────
    logger.info(
        f"[speech-check] rms={rms:.5f} (>= {rms_threshold_to_use} ? {rms_ok})  "
        f"pause_ratio={pause_ratio:.3f} (<= {MAX_PAUSE_RATIO} ? {pause_ok})  "
        f"speech_rate={speech_rate:.2f} (>= {MIN_SPEECH_RATE} ? {rate_ok})  "
        f"strong_indicators={strong_speech_indicators}  "
        f"=> has_speech={has_speech}"
    )
    
    if has_speech:
        logger.info("[speech-check] ACCEPTED — audio contains sufficient speech characteristics.")
    else:
        logger.warning(
            f"[speech-check] REJECTED — failed conditions: "
            f"{[k for k, v in checks.items() if not v]}"
        )

    return {"has_speech": has_speech, "reasons": reasons}


def _extract_features_inline(audio_signal, sample_rate):
    """Inline fallback feature extraction matching heet_audio_module logic."""
    import librosa
    import numpy as np

    # Pitch
    pitches = librosa.yin(audio_signal, fmin=80, fmax=400, sr=sample_rate)
    pitch_mean = float(np.mean(pitches))
    pitch_std = float(np.std(pitches))

    # Energy
    energy = float(np.sum(audio_signal ** 2) / len(audio_signal))

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))

    # Pause ratio
    rms = librosa.feature.rms(y=audio_signal, frame_length=2048, hop_length=512)[0]
    threshold = np.mean(rms) * 0.5
    silent_frames = np.sum(rms < threshold)
    pause_ratio = float(silent_frames / len(rms)) if len(rms) > 0 else 0.0

    # Speech rate
    onset_frames = librosa.onset.onset_detect(y=audio_signal, sr=sample_rate, hop_length=512)
    duration_seconds = len(audio_signal) / sample_rate
    speech_rate = float(len(onset_frames) / duration_seconds) if duration_seconds > 0 else 0.0

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy": energy,
        "mfcc_mean": mfcc_mean,
        "pause_ratio": pause_ratio,
        "speech_rate": speech_rate
    }
