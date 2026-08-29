import os
import subprocess
import streamlit as st
import ollama
from faster_whisper import WhisperModel

# --- Configuration ---
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

# --- App Logic ---

@st.cache_resource
def load_whisper():
    """Cache the Whisper model so it doesn't reload on every interaction."""
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

def transcribe(whisper, audio_path):
    segments, _ = whisper.transcribe(audio_path, language="en", vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments)

def generate_audio(text):
    command = [
        "piper",
        "--model",
        PIPER_MODEL,
        "--output_file",
        RESPONSE_AUDIO_FILE,
    ]
    process = subprocess.run(command, input=text, text=True, capture_output=True)
    
    if process.returncode != 0:
        st.error(f"Piper error: {process.stderr}")
        return False
    return True

# --- UI Layout ---

def main():
    st.set_page_config(page_title="Articulate AI", page_icon="🗣️")
    st.title("🗣️ Articulate AI Coach")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    whisper = load_whisper()

    # Display previous conversation (skip the system prompt)
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Streamlit's native audio recorder (allows you to start/stop whenever you want)
    audio_value = st.audio_input("Record your message")

    if audio_value is not None:
        # 1. Save web audio to file
        with open(AUDIO_FILE, "wb") as f:
            f.write(audio_value.getbuffer())

        # 2. Transcribe
        with st.spinner("🧠 Listening to what you said..."):
            user_text = transcribe(whisper, AUDIO_FILE)

        if not user_text.strip():
            st.warning("⚠️ I couldn't hear anything. Please try again.")
            return

        # Display user message and save to history
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # 3. Get AI response using the whole conversation history
        with st.spinner("🤖 Qwen is thinking..."):
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=st.session_state.messages
            )
            ai_text = response["message"]["content"]

        # Display AI message and save to history
        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        with st.chat_message("assistant"):
            st.write(ai_text)

        # 4. Generate & Play Audio
        with st.spinner("🔊 Generating speech..."):
            success = generate_audio(ai_text)
            if success and os.path.exists(RESPONSE_AUDIO_FILE):
                # Play directly in the browser automatically
                st.audio(RESPONSE_AUDIO_FILE, format="audio/wav", autoplay=True)

if __name__ == "__main__":
    main()