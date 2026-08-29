import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import ollama
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly

from faster_whisper import WhisperModel

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import uvicorn


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = "medium"
OLLAMA_MODEL = "phi3"
PIPER_MODEL = "en_US-lessac-medium.onnx"

# Use None for default device to allow sounddevice to pick the system default
MICROPHONE_DEVICE = None

NATIVE_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000

OUTPUT_DIR = Path("articulate_audio")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# ARTICULATION COACH PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an English conversation partner and articulation coach.

The user understands English well but sometimes struggles to
express thoughts spontaneously, especially during meetings.

Your goal is to help the user become clearer, more fluent,
confident, natural, and spontaneous in spoken English.

Rules:

- Behave like a real conversation partner.
- Ask ONE relevant follow-up question.
- Keep responses concise because they will be spoken aloud.
- Do not behave like a grammar textbook.
- Do not correct every small mistake.
- Do not give long explanations during normal conversation.
- Let the user finish their thought.
- Encourage the user to continue speaking.
- Use natural conversational English.
- If the user's meaning is understandable, continue the
  conversation instead of correcting every grammar mistake.
- Occasionally challenge the user's explanation.
- The goal is spontaneous articulation and confidence.
"""


# ============================================================
# LOAD WHISPER
# ============================================================

print()
print("=" * 60)
print("ARTICULATE AI")
print("=" * 60)
print("Loading Whisper medium...")

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)

print("Whisper ready.")


# ============================================================
# MICROPHONE RECORDER
# ============================================================

class MicrophoneRecorder:

    def __init__(self):

        self.lock = threading.Lock()

        self.recording = False

        self.frames = []

        self.stream = None

    def callback(
        self,
        indata,
        frames,
        time,
        status,
    ):

        if status:
            print("Microphone:", status)

        with self.lock:

            if self.recording:

                self.frames.append(
                    indata.copy()
                )

    def start(self):

        with self.lock:

            if self.recording:
                return False

            self.frames = []
            self.recording = True

        try:

            self.stream = sd.InputStream(
                samplerate=NATIVE_SAMPLE_RATE,
                device=MICROPHONE_DEVICE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self.callback,
            )

            self.stream.start()

            print()
            print("=" * 60)
            print("🎙️ RECORDING STARTED")
            print("=" * 60)

            return True

        except Exception as error:

            with self.lock:
                self.recording = False

            print(
                "Microphone error:",
                error,
            )

            return False

    def stop(self):

        with self.lock:

            if not self.recording:
                return None

            self.recording = False

        print()
        print("=" * 60)
        print("⏹️ RECORDING STOPPED")
        print("=" * 60)

        if self.stream:

            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        with self.lock:

            frames = self.frames
            self.frames = []

        if not frames:
            return None

        return np.concatenate(
            frames,
            axis=0,
        )


recorder = MicrophoneRecorder()


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation = []


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe_audio(audio):
    # Just save the raw audio to check if it's working
    native_path = OUTPUT_DIR / "recording_44100.wav"
    audio_int16 = (audio * 32767).astype(np.int16)
    write(str(native_path), NATIVE_SAMPLE_RATE, audio_int16)

    # SIMPLIFIED PIPELINE: Remove all complex filters
    mono = audio[:, 0] if audio.ndim > 1 else audio
    
    # Basic Resample
    resampled = resample_poly(mono, WHISPER_SAMPLE_RATE, NATIVE_SAMPLE_RATE)
    resampled_int16 = (resampled * 32767).astype(np.int16)
    
    whisper_path = OUTPUT_DIR / "recording_16000.wav"
    write(str(whisper_path), WHISPER_SAMPLE_RATE, resampled_int16)

    print("🧠 Whisper transcribing...")
    segments, info = whisper.transcribe(
        str(whisper_path),
        language="en",
        beam_size=1,
        temperature=0,
        vad_filter=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    
    print()
    print("📝 YOU SAID:")
    print(text)
    print()
    
    return text


# ============================================================
# QWEN
# ============================================================

def ask_qwen(user_text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        conversation[-12:]
    )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    print("🤖 Qwen thinking...")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )

    ai_text = (
        response["message"]["content"]
        .strip()
    )

    print()
    print("🤖 AI:")
    print(ai_text)
    print()

    return ai_text


# ============================================================
# PIPER
# ============================================================

def generate_speech(text):
    
    filename = (
        OUTPUT_DIR /
        "ai_response.wav"
    )
    
    result = subprocess.run(
        [
            "piper",
            "--model",
            PIPER_MODEL,
            "--output_file",
            str(filename),
        ],
        input=text,
        text=True,
        capture_output=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )
    
    return filename



# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Articulate AI"
)


# ============================================================
# HTML UI
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
/>

<title>Articulate AI</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    min-height: 100vh;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top,
            #eef4ff 0%,
            #f8fafc 45%,
            #ffffff 100%
        );

    color: #172033;
}


.container {

    width: min(
        900px,
        calc(100% - 32px)
    );

    margin: 0 auto;

    padding: 40px 0 60px;
}


.header {

    text-align: center;

    margin-bottom: 32px;
}


.logo {

    width: 68px;

    height: 68px;

    border-radius: 20px;

    margin: 0 auto 16px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 32px;

    background: #172033;

    color: white;

    box-shadow:
        0 12px 30px
        rgba(23, 32, 51, .15);
}


h1 {

    margin: 0;

    font-size: 36px;

    letter-spacing: -1px;
}


.subtitle {

    margin-top: 10px;

    color: #667085;

    font-size: 16px;
}


.badges {

    display: flex;

    justify-content: center;

    gap: 8px;

    flex-wrap: wrap;

    margin-top: 18px;
}


.badge {

    padding: 7px 12px;

    border-radius: 999px;

    background: white;

    border: 1px solid #e4e7ec;

    color: #667085;

    font-size: 13px;
}


/* ---------------------------------------------------------
   AI CARD
--------------------------------------------------------- */

.ai-card {

    background: white;

    border:
        1px solid #e4e7ec;

    border-radius: 24px;

    padding: 26px;

    box-shadow:
        0 12px 40px
        rgba(16, 24, 40, .06);

    margin-bottom: 22px;
}


.ai-label {

    color: #667085;

    font-size: 13px;

    font-weight: 600;

    margin-bottom: 10px;

    text-transform: uppercase;

    letter-spacing: .06em;
}


.ai-text {

    font-size: 22px;

    line-height: 1.5;

    font-weight: 500;
}


.audio-wrapper {

    margin-top: 20px;
}


audio {

    width: 100%;
}


/* ---------------------------------------------------------
   RECORD AREA
--------------------------------------------------------- */

.record-card {

    background: white;

    border:
        1px solid #e4e7ec;

    border-radius: 24px;

    padding: 32px;

    text-align: center;

    box-shadow:
        0 12px 40px
        rgba(16, 24, 40, .06);
}


.record-button {

    width: 116px;

    height: 116px;

    border-radius: 50%;

    border: none;

    cursor: pointer;

    font-size: 42px;

    background: #172033;

    color: white;

    box-shadow:
        0 15px 35px
        rgba(23, 32, 51, .25);

    transition:
        transform .15s ease,
        background .2s ease;
}


.record-button:hover {

    transform: scale(1.04);
}


.record-button.recording {

    background: #e5484d;

    animation:
        pulse 1.5s infinite;
}


@keyframes pulse {

    0% {
        box-shadow:
            0 0 0 0
            rgba(229, 72, 77, .45);
    }

    70% {
        box-shadow:
            0 0 0 22px
            rgba(229, 72, 77, 0);
    }

    100% {
        box-shadow:
            0 0 0 0
            rgba(229, 72, 77, 0);
    }
}


.record-title {

    margin-top: 20px;

    font-size: 18px;

    font-weight: 700;
}


.record-hint {

    margin-top: 8px;

    color: #667085;
}


.timer {

    margin-top: 14px;

    font-variant-numeric: tabular-nums;

    color: #e5484d;

    font-weight: 700;
}


/* ---------------------------------------------------------
   TRANSCRIPT
--------------------------------------------------------- */

.transcript-card {

    background: white;

    border:
        1px solid #e4e7ec;

    border-radius: 20px;

    padding: 22px;

    margin-top: 22px;
}


.section-label {

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: .06em;

    color: #667085;

    font-weight: 700;

    margin-bottom: 10px;
}


.transcript {

    font-size: 17px;

    line-height: 1.65;

    color: #344054;
}


/* ---------------------------------------------------------
   CONVERSATION
--------------------------------------------------------- */

.conversation {

    margin-top: 28px;
}


.message {

    padding: 18px 20px;

    border-radius: 18px;

    margin-bottom: 12px;

    line-height: 1.55;
}


.message.user {

    background: #eef4ff;

    margin-left: 80px;
}


.message.ai {

    background: white;

    border:
        1px solid #e4e7ec;

    margin-right: 80px;
}


.message-label {

    font-size: 12px;

    font-weight: 700;

    color: #667085;

    margin-bottom: 7px;

    text-transform: uppercase;
}


/* ---------------------------------------------------------
   STATUS
--------------------------------------------------------- */

.status {

    text-align: center;

    margin-top: 20px;

    color: #667085;

    font-size: 14px;
}


.clear {

    display: block;

    margin: 26px auto 0;

    border: none;

    background: transparent;

    color: #667085;

    cursor: pointer;
}


.clear:hover {

    color: #172033;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 650px) {

    .container {

        width:
            calc(100% - 20px);

        padding-top: 24px;
    }


    h1 {

        font-size: 30px;
    }


    .ai-text {

        font-size: 19px;
    }


    .record-card {

        padding: 24px;
    }


    .message.user {

        margin-left: 20px;
    }


    .message.ai {

        margin-right: 20px;
    }
}

</style>

</head>


<body>


<div class="container">


    <!-- HEADER -->

    <div class="header">

        <div class="logo">
            🎙️
        </div>

        <h1>
            Articulate AI
        </h1>

        <div class="subtitle">
            Your private English speaking partner
        </div>

        <div class="badges">

            <div class="badge">
                🧠 Qwen 2.5 7B
            </div>

            <div class="badge">
                👂 Whisper Small
            </div>

            <div class="badge">
                🔊 Piper
            </div>

            <div class="badge">
                🔒 Local
            </div>

        </div>

    </div>


    <!-- AI -->

    <div class="ai-card">

        <div class="ai-label">
            Your conversation partner
        </div>

        <div
            id="aiText"
            class="ai-text"
        >
            Hello! Let's practice.
            Tell me about something you
            worked on recently.
        </div>

        <div
            id="audioWrapper"
            class="audio-wrapper"
            style="display:none"
        >

            <audio
                id="aiAudio"
                controls
            ></audio>

        </div>

    </div>


    <!-- RECORD -->

    <div class="record-card">

        <button
            id="recordButton"
            class="record-button"
            onclick="toggleRecording()"
        >
            🎙️
        </button>

        <div
            id="recordTitle"
            class="record-title"
        >
            Tap to speak
        </div>

        <div
            id="recordHint"
            class="record-hint"
        >
            Speak naturally. Tap again when you're finished.
        </div>

        <div
            id="timer"
            class="timer"
        ></div>

    </div>


    <!-- STATUS -->

    <div
        id="status"
        class="status"
    >
        Ready when you are.
    </div>


    <!-- TRANSCRIPT -->

    <div
        id="transcriptCard"
        class="transcript-card"
        style="display:none"
    >

        <div class="section-label">
            What I heard
        </div>

        <div
            id="transcript"
            class="transcript"
        ></div>

    </div>


    <!-- CONVERSATION -->

    <div
        id="conversation"
        class="conversation"
    ></div>


    <button
        class="clear"
        onclick="clearConversation()"
    >
        Clear conversation
    </button>


</div>


<script>

let recording = false;

let timerInterval = null;

let seconds = 0;


/* ---------------------------------------------------------
   TOGGLE
--------------------------------------------------------- */

async function toggleRecording() {

    if (!recording) {

        await startRecording();

    } else {

        await stopRecording();

    }
}


/* ---------------------------------------------------------
   START
--------------------------------------------------------- */

async function startRecording() {

    setStatus(
        "🔴 Listening... speak naturally."
    );

    const response = await fetch(
        "/record/start",
        {
            method: "POST"
        }
    );

    const data = await response.json();

    if (!data.success) {

        setStatus(
            "❌ Could not start microphone."
        );

        return;
    }

    recording = true;

    const button =
        document.getElementById(
            "recordButton"
        );

    button.classList.add(
        "recording"
    );

    button.innerText = "⏹️";

    document.getElementById(
        "recordTitle"
    ).innerText =
        "Listening...";

    document.getElementById(
        "recordHint"
    ).innerText =
        "Speak naturally. Tap again when you're finished.";

    seconds = 0;

    updateTimer();

    timerInterval = setInterval(
        () => {

            seconds++;

            updateTimer();

        },
        1000
    );
}


/* ---------------------------------------------------------
   STOP
--------------------------------------------------------- */

async function stopRecording() {

    recording = false;

    clearInterval(
        timerInterval
    );

    const button =
        document.getElementById(
            "recordButton"
        );

    button.classList.remove(
        "recording"
    );

    button.innerText = "⏳";

    document.getElementById(
        "recordTitle"
    ).innerText =
        "Processing...";

    document.getElementById(
        "recordHint"
    ).innerText =
        "Listening → understanding → thinking → speaking";


    setStatus(
        "🧠 Understanding what you said..."
    );


    const response = await fetch(
        "/record/stop",
        {
            method: "POST"
        }
    );


    const data = await response.json();


    if (!data.success) {

        button.innerText = "🎙️";

        document.getElementById(
            "recordTitle"
        ).innerText =
            "Tap to speak";

        setStatus(
            "❌ " + data.error
        );

        return;
    }


    /* -----------------------------------------------------
       TRANSCRIPT
    ----------------------------------------------------- */

    document.getElementById(
        "transcriptCard"
    ).style.display = "block";

    document.getElementById(
        "transcript"
    ).innerText =
        data.transcript;


    setStatus(
        "🤖 AI is thinking..."
    );


    /* -----------------------------------------------------
       AI TEXT
    ----------------------------------------------------- */

    document.getElementById(
        "aiText"
    ).innerText =
        data.ai_text;


    /* -----------------------------------------------------
       AUDIO
    ----------------------------------------------------- */

    if (data.audio_url) {

        const audio =
            document.getElementById(
                "aiAudio"
            );

        audio.src =
            data.audio_url +
            "?t=" +
            Date.now();

        document.getElementById(
            "audioWrapper"
        ).style.display =
            "block";

        audio.play().catch(
            () => {
                console.log(
                    "Browser blocked autoplay."
                );
            }
        );
    }


    renderConversation(
        data.conversation
    );


    button.innerText = "🎙️";

    document.getElementById(
        "recordTitle"
    ).innerText =
        "Tap to speak";

    document.getElementById(
        "recordHint"
    ).innerText =
        "Speak naturally. Tap again when you're finished.";


    setStatus(
        "Ready for your next turn."
    );
}


/* ---------------------------------------------------------
   TIMER
--------------------------------------------------------- */

function updateTimer() {

    const minutes =
        Math.floor(seconds / 60);

    const secs =
        seconds % 60;

    document.getElementById(
        "timer"
    ).innerText =
        String(minutes).padStart(2, "0")
        + ":"
        + String(secs).padStart(2, "0");
}


/* ---------------------------------------------------------
   STATUS
--------------------------------------------------------- */

function setStatus(text) {

    document.getElementById(
        "status"
    ).innerText =
        text;
}


/* ---------------------------------------------------------
   CONVERSATION
--------------------------------------------------------- */

function renderConversation(
    messages
) {

    const container =
        document.getElementById(
            "conversation"
        );

    container.innerHTML = "";


    messages.forEach(
        message => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "message " +
                (
                    message.role === "user"
                    ? "user"
                    : "ai"
                );


            const label =
                document.createElement(
                    "div"
                );

            label.className =
                "message-label";

            label.innerText =
                message.role === "user"
                ? "You"
                : "AI";


            const content =
                document.createElement(
                    "div"
                );

            content.innerText =
                message.content;


            div.appendChild(
                label
            );

            div.appendChild(
                content
            );

            container.appendChild(
                div
            );
        }
    );
}


/* ---------------------------------------------------------
   CLEAR
--------------------------------------------------------- */

async function clearConversation() {

    await fetch(
        "/conversation/clear",
        {
            method: "POST"
        }
    );

    document.getElementById(
        "conversation"
    ).innerHTML = "";

    document.getElementById(
        "transcriptCard"
    ).style.display = "none";

    document.getElementById(
        "aiText"
    ).innerText =
        "Hello! Let's practice. Tell me about something you worked on recently.";

    document.getElementById(
        "audioWrapper"
    ).style.display = "none";

    setStatus(
        "Ready when you are."
    );
}

</script>


</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    return HTML


# ============================================================
# START RECORDING API
# ============================================================

@app.post("/record/start")
def start_recording():

    success = recorder.start()

    return {
        "success": success
    }


# ============================================================
# STOP + PROCESS
# ============================================================

@app.post("/record/stop")
def stop_recording():

    global conversation

    audio = recorder.stop()

    if audio is None:

        return JSONResponse(
            {
                "success": False,
                "error": "No audio was recorded.",
            }
        )


    try:

        # ----------------------------------------------
        # WHISPER
        # ----------------------------------------------

        user_text = transcribe_audio(
            audio
        )

        if not user_text:

            return JSONResponse(
                {
                    "success": False,
                    "error": "I couldn't understand your speech.",
                }
            )


        # ----------------------------------------------
        # QWEN
        # ----------------------------------------------

        ai_text = ask_qwen(
            user_text
        )


        # ----------------------------------------------
        # SAVE CONVERSATION
        # ----------------------------------------------

        conversation.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        conversation.append(
            {
                "role": "assistant",
                "content": ai_text,
            }
        )


        # ----------------------------------------------
        # PIPER
        # ----------------------------------------------

        audio_file = generate_speech(
            ai_text
        )


        return {
            "success": True,
            "transcript": user_text,
            "ai_text": ai_text,
            "audio_url":
                "/audio/" +
                audio_file.name,
            "conversation":
                conversation,
        }


    except Exception as error:

        print()
        print("=" * 60)
        print("PROCESSING ERROR")
        print("=" * 60)
        print(error)
        print()

        return JSONResponse(
            {
                "success": False,
                "error": str(error),
            }
        )


# ============================================================
# AUDIO FILE
# ============================================================

@app.get(
    "/audio/{filename}"
)
def audio(filename):

    path = (
        OUTPUT_DIR /
        filename
    )

    if not path.exists():

        return JSONResponse(
            {
                "error": "Audio not found"
            },
            status_code=404,
        )

    return FileResponse(
        path,
        media_type="audio/wav",
    )


# ============================================================
# CLEAR CONVERSATION
# ============================================================

@app.post(
    "/conversation/clear"
)
def clear():

    global conversation

    conversation = []

    return {
        "success": True
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("--- ARTICULATE AI ---")
    print("=" * 60)
    print()
    print("Open:")
    print()
    print("http://127.0.0.1:8000")
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )