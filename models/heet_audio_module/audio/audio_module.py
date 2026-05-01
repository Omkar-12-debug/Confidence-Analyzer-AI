import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


class AudioRecorder:
    """
    Handles microphone audio recording for the project.
    """

    def __init__(self, duration=5, sample_rate=22050):
        """
        Initialize recording parameters.

        Parameters:
        duration (int): recording time in seconds
        sample_rate (int): audio sampling rate
        """
        self.duration = duration
        self.sample_rate = sample_rate

    def record_audio(self):
        """
        Record audio from microphone.

        Returns:
        numpy array: flattened audio waveform
        """

        print("\nRecording started... Speak now.")

        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )

        sd.wait()

        print("Recording finished.\n")

        # Convert to 1D waveform
        audio_signal = audio.flatten()

        return audio_signal

    def save_audio(self, audio_signal, filename="recorded_audio.wav"):
        """
        Save recorded audio to file.

        Parameters:
        audio_signal (numpy array): recorded waveform
        filename (str): output filename
        """

        write(filename, self.sample_rate, audio_signal)
        print(f"Audio saved as {filename}")


# -------------------------
# Test execution block
# -------------------------
if __name__ == "__main__":

    recorder = AudioRecorder(duration=5)

    audio_signal = recorder.record_audio()

    print("Audio signal length:", len(audio_signal))

    print("First 10 waveform values:")
    print(audio_signal[:10])

    recorder.save_audio(audio_signal)