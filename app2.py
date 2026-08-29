# import sounddevice as sd
# from scipy.io.wavfile import write
# from faster_whisper import WhisperModel
# import ollama

# SAMPLE_RATE = 16000
# DURATION = 10
# AUDIO_FILE = "my_voice.wav"
# OLLAMA_MODEL = "qwen2.5:7b"


# def record_voice():
#     print("\n🎙️ Speak now...")
#     print("Recording for 10 seconds...\n")

#     audio = sd.rec(
#         int(DURATION * SAMPLE_RATE),
#         samplerate=SAMPLE_RATE,
#         channels=1,
#         dtype="int16",
#     )

#     sd.wait()

#     write(AUDIO_FILE, SAMPLE_RATE, audio)

#     print("✅ Recording finished.")


# def transcribe(model):
#     print("🧠 Understanding what you said...")

#     segments, info = model.transcribe(
#         AUDIO_FILE,
#         language="en",
#         vad_filter=True,
#     )

#     text = " ".join(segment.text.strip() for segment in segments)

#     return text


# def ask_qwen(text):
#     print("\n🤖 Qwen is thinking...\n")

#     response = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": """
# You are an English conversation partner and articulation coach.

# The user understands English but struggles to articulate thoughts
# spontaneously, especially during meetings.

# For now, behave primarily as a natural conversation partner.

# Rules:
# - Respond naturally.
# - Ask one relevant follow-up question.
# - Do not correct every mistake.
# - Do not give grammar lectures.
# - Keep your response reasonably short.
# - Encourage the user to continue speaking.
# """,
#             },
#             {
#                 "role": "user",
#                 "content": text,
#             },
#         ],
#     )

#     return response["message"]["content"]


# def main():
#     print("Loading Whisper...")
#     whisper = WhisperModel(
#         "small",
#         device="cpu",
#         compute_type="int8",
#     )

#     print("\n===================================")
#     print("      ARTICULATE AI - TEST")
#     print("===================================")

#     record_voice()

#     text = transcribe(whisper)

#     print("\n👤 You said:")
#     print("-----------------------------------")
#     print(text)
#     print("-----------------------------------")

#     response = ask_qwen(text)

#     print("🤖 Qwen:")
#     print("-----------------------------------")
#     print(response)
#     print("-----------------------------------")


# if __name__ == "__main__":
#     main()