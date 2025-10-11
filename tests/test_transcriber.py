# Run with:
#   python -m tests.test_transcriber
# or:
#   pytest -q tests/test_transcriber.py

"""
Transcriber (faster-whisper) test.

- Instantiates Transcriber
- Runs transcription on a sample WAV
- Writes segments + metadata to data/processed/test_transcriber/<timestamped>.json
"""

from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import logging
import json
import time
import os

from audio_utils.transcriber import Transcriber

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE = "sample_1_9m"
ROOT = Path(__file__).resolve().parents[1]  # repo root
AUDIO_PATH = ROOT / "data" / "raw" / f"{SAMPLE}.wav"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = ROOT / "data" / "processed" / "test_transcriber"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUT_DIR / f"test_transcriber_output_{SAMPLE}_{TIMESTAMP}.json"

def test_transcription():
    """Run transcription on SAMPLE and write result to OUTPUT_PATH."""
    logger.info("Starting Transcription Test")

    # Load model
    transcriber = Transcriber(model_size="medium")  # Other options: "medium" or "base"

    # Run transcription
    start_time = time.time()
    result = transcriber.transcribe(AUDIO_PATH, word_timestamps=True)
    end_time = time.time()

    logger.info("Transcription completed in %ss", round(end_time - start_time, 2))
    logger.info("Found %d transcript segments", len(result['segments']))

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Output written to: %s", OUTPUT_PATH)

if __name__ == "__main__":
    test_transcription()