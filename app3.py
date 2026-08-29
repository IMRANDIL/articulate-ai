import os
import subprocess

import ollama
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
DURATION = 10

AUDIO_FILE = "my_voice.wav"
RESPONSE_AUDIO_FILE = "ai_response.wav"

WHISPER_MODEL = "small"
OLLAMA_MODEL = "qwen2.5:7b"
PIPER_MODEL = "en_US-lessac-medium.onnx"


SYSTEM_PROMPT = """
You are a natural English conversation partner and articulation coach.

The user understands English but struggles to express thoughts
spontaneously, especially in meetings.

Your primary job is to have a natural spoken conversation.

Rules:
- Talk like a real conversation partner.
- Ask one relevant follow-up question.
- Keep responses short enough for spoken conversation.
- Do not give grammar lectures.
- Do not correct every mistake.
- Do not interrupt the user's flow.
- Encourage the user to keep talking.
"""


def record_voice():
    print("\n🎙️ Your turn...")
    print("Speak for 10 seconds...\n")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    write(AUDIO_FILE, SAMPLE_RATE, audio)

    print("✅ Recording finished.")


def transcribe(whisper):
    print("🧠 Listening to what you said...")

    segments, _ = whisper.transcribe(
        AUDIO_FILE,
        language="en",
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text


def ask_qwen(text):
    print("🤖 Qwen is thinking...")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    return response["message"]["content"]


def speak(text):
    print("🔊 AI is speaking...")

    command = [
        "piper",
        "--model",
        PIPER_MODEL,
        "--output_file",
        RESPONSE_AUDIO_FILE,
    ]

    process = subprocess.run(
        command,
        input=text,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        print("❌ Piper error:")
        print(process.stderr)
        return

    # Play the WAV using Windows Media Player/default application
    os.startfile(RESPONSE_AUDIO_FILE)


def main():

    print("Loading Whisper...")
    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
    )

    print("\n===================================")
    print("       ARTICULATE AI")
    print("===================================")

    while True:

        record_voice()

        user_text = transcribe(whisper)

        if not user_text.strip():
            print("⚠️ I couldn't hear anything.")
            continue

        print("\n👤 YOU:")
        print("-----------------------------------")
        print(user_text)
        print("-----------------------------------")

        ai_response = ask_qwen(user_text)

        print("\n🤖 AI:")
        print("-----------------------------------")
        print(ai_response)
        print("-----------------------------------")

        speak(ai_response)

        print("\n-----------------------------------")
        print("Press Ctrl+C to stop.")
        print("-----------------------------------")


if __name__ == "__main__":
    main()