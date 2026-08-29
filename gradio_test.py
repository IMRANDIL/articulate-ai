import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
import gradio as gr


# ============================================================
# CONFIG
# ============================================================

DEVICE = 1
NATIVE_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000
RECORD_SECONDS = 10

AUDIO_NATIVE = "recording_44100.wav"
AUDIO_WHISPER = "recording_16000.wav"

WHISPER_MODEL = "small"


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper Small...")

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)

print("Whisper ready.")


# ============================================================
# RECORD
# ============================================================

def record_audio():

    print("\n🎙️ Recording...")
    print("Speak naturally...")

    audio = sd.rec(
        int(RECORD_SECONDS * NATIVE_SAMPLE_RATE),
        samplerate=NATIVE_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=DEVICE,
    )

    sd.wait()

    print("✅ Recording finished.")

    # Convert float [-1, 1] to int16
    audio_int16 = np.clip(
        audio * 32767,
        -32768,
        32767,
    ).astype(np.int16)

    write(
        AUDIO_NATIVE,
        NATIVE_SAMPLE_RATE,
        audio_int16,
    )

    return audio


# ============================================================
# RESAMPLE
# ============================================================

def convert_to_whisper_rate(audio):

    print(
        f"🔄 Resampling "
        f"{NATIVE_SAMPLE_RATE} Hz → "
        f"{WHISPER_SAMPLE_RATE} Hz..."
    )

    # 44100 -> 16000
    resampled = resample_poly(
        audio[:, 0],
        WHISPER_SAMPLE_RATE,
        NATIVE_SAMPLE_RATE,
    )

    resampled_int16 = np.clip(
        resampled * 32767,
        -32768,
        32767,
    ).astype(np.int16)

    write(
        AUDIO_WHISPER,
        WHISPER_SAMPLE_RATE,
        resampled_int16,
    )

    return AUDIO_WHISPER


# ============================================================
# RECORD + TRANSCRIBE
# ============================================================

def record_and_transcribe():

    audio = record_audio()

    whisper_audio = convert_to_whisper_rate(
        audio
    )

    print("🧠 Transcribing...")

    segments, info = whisper.transcribe(
        whisper_audio,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
        beam_size=5,
    )

    segments = list(segments)

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    print("\n==============================")
    print("YOU SAID:")
    print(text)
    print("==============================")

    return (
        AUDIO_NATIVE,
        AUDIO_WHISPER,
        text,
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Articulate AI"
) as demo:

    gr.Markdown(
        """
        # 🎙️ Articulate AI — Microphone Diagnostic

        This test records your Windows microphone at its
        native 44.1 kHz rate and then converts it to 16 kHz
        for Whisper.
        """
    )

    record_button = gr.Button(
        "🎙️ Start Speaking",
        variant="primary",
    )

    native_audio = gr.Audio(
        label="🎧 Original 44.1 kHz Recording",
        type="filepath",
    )

    whisper_audio = gr.Audio(
        label="🎧 16 kHz Audio Sent to Whisper",
        type="filepath",
    )

    transcript = gr.Textbox(
        label="📝 Whisper Heard",
        lines=5,
    )

    record_button.click(
        fn=record_and_transcribe,
        outputs=[
            native_audio,
            whisper_audio,
            transcript,
        ],
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    demo.launch()