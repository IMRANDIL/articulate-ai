import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 10
AUDIO_FILE = "my_voice.wav"

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print("\n🎙️ Recording for 10 seconds...")
print("Speak naturally...\n")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
)

sd.wait()

write(AUDIO_FILE, SAMPLE_RATE, audio)

print("✅ Recording finished.")
print("🧠 Transcribing...\n")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="en",
    vad_filter=True,
)

text = " ".join(segment.text.strip() for segment in segments)

print("You said:")
print("-" * 50)
print(text)
print("-" * 50)