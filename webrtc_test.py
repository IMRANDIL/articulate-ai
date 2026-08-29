import av
import streamlit as st

from streamlit_webrtc import webrtc_streamer


st.set_page_config(
    page_title="Voice Test",
    page_icon="🎙️",
)

st.title("🎙️ Articulate AI — Voice Test")

st.write(
    "Click START and allow microphone access."
)

webrtc_streamer(
    key="voice-test",
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
)