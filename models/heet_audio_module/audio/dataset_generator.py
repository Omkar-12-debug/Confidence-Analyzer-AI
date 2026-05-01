import csv
import os

from audio.audio_module import AudioRecorder
from audio.feature_extraction import extract_all_features


DATASET_PATH = "dataset/audio_dataset.csv"


def initialize_dataset():
    """
    Create dataset file with header if it doesn't exist
    """

    if not os.path.exists(DATASET_PATH):

        with open(DATASET_PATH, mode='w', newline='') as file:
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

        print("Dataset file created.")


def append_to_dataset(features, label):

    with open(DATASET_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            features["pitch_mean"],
            features["pitch_std"],
            features["energy"],
            features["mfcc_mean"],
            features["pause_ratio"],
            features["speech_rate"],
            label
        ])


def main():

    initialize_dataset()

    recorder = AudioRecorder(duration=5)

    print("\nAnswer the interview question clearly.")

    audio_signal = recorder.record_audio()

    features = extract_all_features(audio_signal, recorder.sample_rate)

    print("\nExtracted Features:")
    print(features)

    label = input("\nEnter label (confident / neutral / nervous): ")

    append_to_dataset(features, label)

    print("\nSample added to dataset.")


if __name__ == "__main__":
    main()