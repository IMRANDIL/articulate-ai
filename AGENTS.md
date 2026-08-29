# AGENTS.md - Articulate AI

## System Overview
Local AI articulation coach using:
- **STT**: faster-whisper (`small` model)
- **LLM**: Ollama (`qwen2.5:7b`)
- **TTS**: Piper (`en_US-lessac-medium.onnx`)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Single-file HTML embedded in `app.py`

## Developer Commands
- **Run App**: `python app.py`
- **Environment**: Uses a local `.venv` (Python 3.12)

## Architecture & Constraints
- **Audio Flow**: 
    1. Record @ 44.1kHz $\rightarrow$ `articulate_audio/recording_44100.wav`
    2. Resample to 16kHz $\rightarrow$ `articulate_audio/recording_16000.wav` $\rightarrow$ Whisper
    3. LLM Response $\rightarrow$ Piper $\rightarrow$ Random UUID `.wav` file in `articulate_audio/`
- **Microphone**: Hardcoded `MICROPHONE_DEVICE = 1` in `app.py:31`. Change this if the mic is not detected.
- **Model Files**: Piper `.onnx` and `.json` files must be in the root directory.
- **State**: Conversation memory is stored in a global Python list (`conversation`) and is reset on app restart or via `/conversation/clear`.
