# Run with:
#   python -m tests.test_audio_processing
# or:
#   pytest -q tests/test_audio_processing.py

"""
Integration-style test for audio_processing.process_audio.

- Loads Transcriber (faster-whisper) and Diarizer (pyannote).
- Processes a sample audio file end-to-end.
- Writes JSON segments + human-readable transcript to data/processed/.
"""

from pipeline.audio_processing import process_audio
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from dotenv import load_dotenv
from datetime import datetime
import torch
import os
import json
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE_NAME = "sample_1_9m"  # change as needed
ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes /tests at repo top level)
SAMPLE_PATH = ROOT / "data" / "raw" / f"{SAMPLE_NAME}.wav"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = ROOT / "data" / "processed" / "test_audio_processing" / f"{SAMPLE_NAME}_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")

def test_audio_processing():
    """ End-to-end processing of SAMPLE_PATH, saving outputs to OUTPUT_DIR. """
    # Load models
    transcriber = Transcriber(model_size="medium") 
    diarizer = Diarizer(hf_token=token)

    # Process audio
    start_time = time.time()
    result = process_audio(transcriber, diarizer, SAMPLE_PATH, job_id="test_job")
    end_time = time.time()
    logger.info("Audio processing completed in %s", end_time - start_time)

    json_transcript = result["json_transcript"]
    complete_transcript = result["complete_transcript"]

    # Save json output
    json_output_path = os.path.join(OUTPUT_DIR, f"audio_processing_output.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_transcript, f, indent=2, ensure_ascii=False)

    logger.info("JSON transcript saved to %s", json_output_path)

    # Save complete transcript for LLM
    transcript_output_path = os.path.join(OUTPUT_DIR, f"audio_processing_transcript.txt")
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        f.write(complete_transcript)

    logger.info("Complete transcript saved to %s", transcript_output_path)

def test_process_audio():
    """ Lightweight smoke test: calls process_audio and asserts the shape of the result."""
    
    # Test the process_audio function with a sample audio file
    transcriber = Transcriber(model_size="medium")
    diarizer = Diarizer(hf_token=token)
    result = process_audio(transcriber, diarizer, SAMPLE_PATH, "test_job")

    assert result is not None
    assert "json_transcript" in result
    assert "complete_transcript" in result
    assert isinstance(result["json_transcript"], list)
    assert isinstance(result["complete_transcript"], str)
    logger.info("process_audio test passed!")

if __name__ == "__main__":
    test_audio_processing()
    test_process_audio()