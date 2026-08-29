# Articulate AI

A local, private AI articulation coach designed to help users improve their spoken English fluency, confidence, and spontaneity.

## 🚀 System Overview
Articulate AI provides a seamless voice-to-voice loop using locally hosted models to ensure privacy and low latency.

- **Speech-to-Text (STT)**: `faster-whisper` (small model) for accurate English transcription.
- **LLM**: `Ollama` (`qwen2.5:7b`) acting as a conversational partner and coach.
- **Text-to-Speech (TTS)**: `Piper` for fast, natural-sounding local voice synthesis.
- **Backend**: `FastAPI` + `Uvicorn`.
- **Frontend**: Minimalist, single-page HTML/JS interface.

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.12**
- **Ollama**: Installed and running with `qwen2.5:7b` pulled.
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
- **Microphone**: If the app cannot detect your microphone, update `MICROPHONE_DEVICE` in `app.py`.
- **Audio Flow**:
    - Native Recording: 44.1kHz
    - Whisper Processing: 16kHz (automatically resampled)
    - Output: Saved to `articulate_audio/` folder.

## 📝 Project Structure
- `app.py`: Main application logic, API endpoints, and embedded HTML frontend.
- `articulate_audio/`: Storage for temporary recording and response files.
- `AGENTS.md`: Developer-specific instructions for AI agents.
- `.gitignore`: Configured to exclude large model files and virtual environments.

## ⚖️ License
Local use only.
