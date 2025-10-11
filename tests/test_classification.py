# Run with:
#   python -m tests.test_classification
# or:
#   pytest -q tests/test_classification.py

"""
Integration-style test for segment classification.

- Loads themes JSON and audio-processing segments JSON from data/processed/.
- Runs Analyzer.classify_all_segments (optionally benchmarks batch sizes).
- Writes classified segments and any failed segments to a timestamped folder.
"""

from text_utils.analyzer import Analyzer
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE_NAME = "sample_1_9m"
ROOT = Path(__file__).resolve().parents[1]  # repo root
THEMES_PATH = ROOT / "data" / "processed" / "test_topic_analysis" / "generated_themes_sample_1_9m.json"
SEGMENTS_PATH = ROOT / "data" / "processed" / "test_audio_processing" / "audio_processing_output.json"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = ROOT / "data" / "processed" / "test_classification" / f"{SAMPLE_NAME}_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "classified_segments.json"

# Load environment (.env) for any required keys (if Analyzer needs them)
load_dotenv()

def test_classification(benchmark_batch_size=False):
    """
    Run classification over precomputed segments with precomputed themes.
    If benchmark_batch_size=True, loop over several batch sizes and skip saving.
    """
    # Load themes from JSON
    try:
        with open(THEMES_PATH, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except Exception as e:
        logger.error("Could not load themes: %s", e)
        return

    # Load segments from JSON
    try:
        with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        logger.error("Could not load segments: %s", e)
        return

    # Initialize LLama model
    logger.info("Loading Analyzer...")
    analyzer = Analyzer()

    if benchmark_batch_size:
        for bsize in [1, 4, 8, 16]:
            logger.info("Benchmarking batch_size=%d", bsize)
            analyzer.classify_all_segments(segments, themes, batch_size=bsize) 
    else:
        logger.info("Classifying each segment")
        segments = analyzer.classify_all_segments(segments, themes)

        # Save output
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        logger.info("Classified segments saved to %s", OUTPUT_PATH)

    # Save failed segments for review
    if analyzer.failed_segments:
        FAILED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"failed_segments.json")
        with open(FAILED_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(analyzer.failed_segments, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d failed segments to %s", len(analyzer.failed_segments), FAILED_OUTPUT_PATH)


if __name__ == "__main__":
    test_classification(benchmark_batch_size=False)


