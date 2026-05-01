import numpy as np
import librosa
try:
    import sounddevice as sd
except ImportError:
    sd = None


# -------------------------------
# Pitch Extraction
# -------------------------------
def calculate_pitch(audio_signal, sample_rate):

    pitches = librosa.yin(
        audio_signal,
        fmin=80,
        fmax=400,
        sr=sample_rate
    )

    pitch_mean = float(np.mean(pitches))
    pitch_std = float(np.std(pitches))

    return pitch_mean, pitch_std


# -------------------------------
# Energy Calculation
# -------------------------------
def calculate_energy(audio_signal):
    """
    Calculate average signal energy.
    """

    energy = np.sum(audio_signal ** 2) / len(audio_signal)

    return energy


# -------------------------------
# MFCC Feature Extraction
# -------------------------------
def calculate_mfcc(audio_signal, sample_rate):
    """
    Extract MFCC features from speech signal.
    """

    mfcc = librosa.feature.mfcc(
        y=audio_signal,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc)

    return mfcc_mean


# -------------------------------
# Pause Ratio Detection
# -------------------------------
def calculate_pause_ratio(audio_signal, sample_rate):

    frame_length = 2048
    hop_length = 512

    # Compute RMS energy for frames
    rms = librosa.feature.rms(
        y=audio_signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Silence threshold
    threshold = np.mean(rms) * 0.5

    silent_frames = np.sum(rms < threshold)

    pause_ratio = silent_frames / len(rms)

    return float(pause_ratio)

# Speec Rate 
def calculate_speech_rate(audio_signal, sample_rate):
    """
    Estimate speech rate using onset detection (approximate syllable rate).
    """

    onset_frames = librosa.onset.onset_detect(
        y=audio_signal,
        sr=sample_rate,
        hop_length=512
    )

    duration_seconds = len(audio_signal) / sample_rate

    if duration_seconds == 0:
        return 0.0

    speech_rate = len(onset_frames) / duration_seconds

    return float(speech_rate)

# -------------------------------
# Combine All Features
# -------------------------------
def extract_all_features(audio_signal, sample_rate):

    pitch_mean, pitch_std = calculate_pitch(audio_signal, sample_rate)

    energy = calculate_energy(audio_signal)

    mfcc_mean = calculate_mfcc(audio_signal, sample_rate)

    pause_ratio = calculate_pause_ratio(audio_signal, sample_rate)

    speech_rate = calculate_speech_rate(audio_signal, sample_rate)

    features = {
        "pitch_mean": float(pitch_mean),
        "pitch_std": float(pitch_std),
        "energy": float(energy),
        "mfcc_mean": float(mfcc_mean),
        "pause_ratio": float(pause_ratio),
        "speech_rate": float(speech_rate)
    }

    return features


# -------------------------------
# Test Block
# -------------------------------
if __name__ == "__main__":

    duration = 5
    sample_rate = 22050

    print("Recording for feature extraction test...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    audio_signal = audio.flatten()

    features = extract_all_features(audio_signal, sample_rate)

    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")