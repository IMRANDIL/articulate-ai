# Articulate AI

A local, private AI articulation coach designed to help users improve their spoken English fluency, confidence, and spontaneity.

## 🚀 The Crux of the Project
Articulate AI implements a fully local **Voice-to-Voice (V2V) loop**. It captures raw audio, converts it to text, processes it through a conversational AI, and speaks the response back to the user—all without sending data to the cloud.

### 🧠 Backend Architecture
The backend is a **FastAPI** server that manages three specialized AI pipelines:
1. **Audio Capture $\rightarrow$ STT**:
   - Captures audio at **44.1kHz**.
   - Uses **`faster-whisper` (Small/Medium)** for high-accuracy, local speech-to-text transcription.
   - Implements a "Raw Mode" pipeline to prevent audio corruption and minimize hallucinations.
2. **Cognition (LLM)**:
   - Uses **`Ollama` (`gemma4:e4b`)** as the articulation coach.
   - Maintains a short-term conversation memory to keep the context of the practice session.
3. **TTS $\rightarrow$ Audio Output**:
   - Uses **`Piper`** (ONNX runtime) for ultra-fast, local text-to-speech synthesis.
   - Outputs fixed-name `.wav` files (`ai_response.wav`) to minimize disk clutter.

### 🎨 Frontend Architecture
The frontend is a **single-file embedded HTML/JS** interface designed for zero-latency interaction:
- **Reactive UI**: Uses a pulse-animation recording button to indicate active listening.
- **Asynchronous Flow**: Leverages the `Fetch API` to trigger recording start/stop and retrieve AI responses without page reloads.
- **Real-time Feedback**: Displays a live transcript of "What I heard" and a conversation history log.

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.12**
- **Ollama**: Installed and running with `gemma4:e4b` pulled.
- **Piper**: Executable and model files (`.onnx` and `.json`) must be in the root directory.

### Quick Start
1. **Clone the repository**
2. **Set up the virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
4. **Access the UI**: Open `http://127.0.0.1:8000` in your browser.

## ⚙️ Configuration
- **Microphone**: Uses system default device. To avoid "dimming" or "vanishing" voice, it is recommended to disable "Audio Enhancements" in Windows Sound Settings.
- **Audio Flow**: 
  - Root storage for audio: `articulate_audio/`
  - Only keeps the current `recording_16000.wav` and `ai_response.wav` to avoid disk clutter.

## 📝 Project Structure
- `app.py`: Core engine (Backend + Embedded Frontend).
- `articulate_audio/`: Temporary audio workspace.
- `AGENTS.md`: AI agent guidance.
- `requirements.txt`: Project dependencies.

## ⚖️ License
Local use only.
