# TalkTrack API 
*Backend for AI-powered interview analysis and automated editing.*

TalkTrack API is the FastAPI-based backend powering the TalkTrack Premiere Pro Plugin, an AI system that analyzes interview footage, detects speakers, extracts key themes, and returns structured annotations for automated editing inside Adobe Premiere Pro.

## Features
- **Speaker Diarization** — Uses pyannote.audio to separate interviewer/interviewee voices.
- **Speech Transcription** — Fast, accurate faster-whisper transcription with word-level timestamps.
- **Topic Extraction & Classification** — Local LLM (Llama 3.1 8B Instruct) identifies key themes and classifies segments.
- **End-to-End Pipeline** — Transcribe → Diarize → Align → Analyze → Classify, all orchestrated in FastAPI background tasks.
- **FastAPI + Async Jobs** — Handles uploads, job polling, and background processing via BackgroundTasks.
- **Structured JSON Output** — Returns segment-level data ready for Premiere marker annotation.
- **Optional Local Testing Suite** — Includes modular test clients for each pipeline stage.

## Tech Stack
| Category | Technology |
|-----------|-----------|
| Framework     | FastAPI    | 
| Speech Processing     | faster-whisper    | 
| Speaker Diarization   | Transformers (Llama 3.1 8B Instruct)  | 
| Environment     | Python 3.10+, CUDA (Opt.)    | 
| Auth     | API key-based    |
| Logging     | Python `logging` + rotating log file    |
| Testing     | Modular test clients under `/tests`    |

## Running Locally
1. Clone the repository
``` bash
git clone https://github.com/yourusername/talktrack_api.git
cd talktrack_api
```
2. Create a virtual environment
``` bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
3. Install dependencies
``` bash
pip install -r requirements.txt
```
4. Set environment variables
``` bash
HF_TOKEN=your_huggingface_token
```
5. Run the API
``` bash
uvicorn api.main:app --reload
```

## Available Endpoints
| Method | Endpoint | Description |
|:------:|-----------|-------------|
| `POST` | `/upload-audio` | Uploads audio, starts background job |
| `GET`  | `/status/{job_id}` | Polls job progress |
| `GET`  | `/results/{job_id}` | Retrieves processed segments |
| `POST` | `/cleanup` | Clears temporary and result files |
| `GET`  | `/ping` | Health check |


## Pipeline stages
Stage	Module	Description
Transcription	audio_utils/transcriber.py	Generates text + word timestamps via Faster-Whisper
Diarization	audio_utils/diarizer.py	Identifies speakers and merges/filters segments
Alignment	pipeline/audio_processing.py	Aligns transcript and speaker segments
Topic Analysis	text_utils/analyzer.py	Extracts and classifies themes using a local Llama model
Export	utils/storage.py, utils/progress.py	Saves structured JSON + job metadata

## Project Status
This is a functional MVP used with the TalkTrack Premiere Pro plugin.
Still under active development — further optimizations, authentication, batching, and UI integration planned.

## Author
Sarah Bethea
B.S. Computer Science (Post-Bacc), University of Montana