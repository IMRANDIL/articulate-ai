from faster_whisper import WhisperModel


AUDIO_FILE = "recording_16000.wav"


models = [
    ("small", "small"),
    ("small.en", "small.en"),
]


for name, model_name in models:

    print("\n")
    print("=" * 60)
    print(f"TESTING WHISPER: {name}")
    print("=" * 60)

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        AUDIO_FILE,
        language="en",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    print("\nTRANSCRIPT:")
    print(text)