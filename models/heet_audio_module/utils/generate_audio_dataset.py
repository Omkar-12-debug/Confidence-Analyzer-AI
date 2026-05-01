import os
import sys
import subprocess
import csv
import librosa

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEET_MODULE_DIR = os.path.join(BASE_DIR, "heet_audio_module")
PARTH_MODULE_DIR = os.path.join(BASE_DIR, "parth_facial_module")
VIDEO_DATA_DIR = os.path.join(BASE_DIR, "video_data")
DATASET_FILE = os.path.join(HEET_MODULE_DIR, "dataset", "audio_dataset.csv")
TEMP_WAV = os.path.join(HEET_MODULE_DIR, "utils", "temp.wav")
LOCAL_FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg-8.1-essentials_build", "bin", "ffmpeg.exe")

# Add paths to sys.path to resolve imports
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# Also add specific modules
if HEET_MODULE_DIR not in sys.path:
    sys.path.append(HEET_MODULE_DIR)

from heet_audio_module.audio.feature_extraction import extract_all_features

def get_ffmpeg_path():
    """Returns the path to ffmpeg executable."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return LOCAL_FFMPEG_PATH

FFMPEG_EXE = get_ffmpeg_path()

def initialize_dataset():
    """Initializes the CSV dataset with headers."""
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    with open(DATASET_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "pitch_mean",
            "pitch_std",
            "energy",
            "mfcc_mean",
            "pause_ratio",
            "speech_rate",
            "label"
        ])
    print("Initialized new dataset file: audio_dataset.csv")

def extract_audio(video_path):
    """Uses ffmpeg to extract mono 16kHz audio from a video."""
    cmd = [
        FFMPEG_EXE,
        "-y",               # Overwrite temp file
        "-i", video_path,
        "-ac", "1",         # Mono channel
        "-ar", "16000",     # 16kHz sample rate
        "-vn",              # No video
        TEMP_WAV
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_predicted_label(video_filename):
    """Predicts confidence label using facial module."""
    return ""

def append_to_dataset(features, label):
    """Appends extracted features and label to the dataset."""
    with open(DATASET_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            features.get("pitch_mean", 0.0),
            features.get("pitch_std", 0.0),
            features.get("energy", 0.0),
            features.get("mfcc_mean", 0.0),
            features.get("pause_ratio", 0.0),
            features.get("speech_rate", 0.0),
            ""
        ])

def main():
    if not os.path.exists(VIDEO_DATA_DIR):
        print(f"Error: Video data directory not found at {VIDEO_DATA_DIR}")
        return

    initialize_dataset()

    videos = sorted([f for f in os.listdir(VIDEO_DATA_DIR) if f.lower().endswith(".mp4")])
    total_processed = 0

    for video in videos:
        # Step 0: Logging
        print(f"Processed: {video}")
        video_path = os.path.join(VIDEO_DATA_DIR, video)

        # Step 1: Extract Audio
        if not extract_audio(video_path):
            print(f"Error processing {video}: FFmpeg audio extraction failed")
            continue

        # Step 2: Extract Features
        try:
            audio_signal, sr = librosa.load(TEMP_WAV, sr=16000, mono=True)
            features = extract_all_features(audio_signal, sr)
        except Exception as e:
            print(f"Error processing {video}: Feature extraction failed - {e}")
            continue

        # Step 3: Assign Label (Empty per requirement)
        label = ""

        # Step 4: Write to CSV
        append_to_dataset(features, label)
        total_processed += 1

    # Cleanup
    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)

    print(f"\nTotal rows written: {total_processed}")

if __name__ == "__main__":
    main()
