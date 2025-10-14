# Run with:
#   python -m tests.test_diarizer
# or:
#   pytest -q tests/test_diarizer.py

"""
Integration-style test for the Diarizer (pyannote.audio).

- Instantiates Diarizer with HF_TOKEN from .env (if present).
- Runs diarization on a sample WAV file.
- Writes cleaned segments to data/processed/test_diarization/<timestamped>.json.
"""

from audio_utils.diarizer import Diarizer
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import time
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE = "sample_1_9m"
ROOT = Path(__file__).resolve().parents[1]  # repo root
AUDIO_PATH = ROOT / "data" / "raw" / f"{SAMPLE}.wav"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = ROOT / "data" / "processed" / "test_diarization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"test_diarizer_output_{SAMPLE}_{TIMESTAMP}.json"

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN") 


def test_diarization():
    """Run diarization on SAMPLE and write cleaned segments to OUTPUT_PATH."""
    # Load model
    diarizer = Diarizer()

    logger.info("Using device %s", diarizer.device.upper())

    # Run diarization
    start_time = time.time()
    result = diarizer.diarize_audio(AUDIO_PATH)
    end_time = time.time()

    logger.info("Diarization completed in %s", round(end_time - start_time, 2))
    logger.info("Found %d cleaned segments from %d raw segments", len(result['segments']), len(result['raw_segments']))

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)

    logger.info("Output written to %s", OUTPUT_PATH)


if __name__ == "__main__":
    test_diarization()