import wave
import numpy as np
from scipy.io.wavfile import write


INPUT = "recording_44100.wav"
OUTPUT = "recording_amplified.wav"

GAIN = 4.0


with wave.open(INPUT, "rb") as wav:

    sample_rate = wav.getframerate()
    frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(
        frames,
        dtype=np.int16,
    ).astype(np.float32)


print("Original peak:", np.max(np.abs(audio)))
print("Original RMS:", np.sqrt(np.mean(audio ** 2)))


# Amplify
audio = audio * GAIN


# Prevent clipping
audio = np.clip(
    audio,
    -32768,
    32767,
)


audio = audio.astype(np.int16)


write(
    OUTPUT,
    sample_rate,
    audio,
)


print("Amplified peak:", np.max(np.abs(audio)))
print("Saved:", OUTPUT)