# Run with:
#   python -m tests.test_topic_analysis
# or:
#   pytest -q tests/test_topic_analysis.py

"""
Topic analysis test.

- Loads a plain-text transcript produced by the audio processing test.
- Runs Analyzer.extract_themes to produce parsed + JSON themes.
- Writes both human-readable and JSON outputs to a timestamped folder.
"""

from datetime import datetime
from pathlib import Path
import logging
import json

from text_utils.analyzer import Analyzer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE_NAME = "sample_1_9m"
ROOT = Path(__file__).resolve().parents[1]  # repo root
TRANSCRIPT_PATH = ROOT / "data" / "processed" / "test_audio_processing" / "audio_processing_transcript.txt"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = ROOT / "data" / "processed" / "test_topic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARSED_OUTPUT_PATH = OUT_DIR / f"generated_themes_{SAMPLE_NAME}_{TIMESTAMP}.txt"
JSON_OUTPUT_PATH = OUT_DIR / f"generated_themes_{SAMPLE_NAME}_{TIMESTAMP}.json"

def test_topic_analysis():  
    """Analyze topics from TRANSCRIPT_PATH and write results to OUT_DIR."""      
    # Read the transcript file
    try:
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            transcript = f.read()
        logger.info("Transcript loaded: %d characters", len(transcript))
    except FileNotFoundError:
        logger.error("Error: Could not find file %s", TRANSCRIPT_PATH)
        return

    # Analyze topics
    logger.info("Initializing topic analyzer...")
    model = Analyzer()

    logger.info("Extracting themes...")
    result = model.extract_themes(transcript)
    logger.info("Found %d themes", len(result['parsed_themes']))

    # Write results to output_path
    with open(PARSED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("TOPIC ANALYSIS RESULTS:\n\n")

        f.write("PARSED THEMES:\n")
        f.write("-" * 20 + "\n")

        for i, theme in enumerate(result["parsed_themes"], 1):
            f.write(f"\n{i}. {theme.get('title', 'Unknown Title')}\n")
            f.write(f"   Description: {theme.get('description', 'No description')}\n")
            f.write(f"   Keywords: {', '.join(theme.get('keywords', []))}\n")


    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result["json_themes"], f, indent=2)

    logger.info("Results written to: %s", JSON_OUTPUT_PATH)

if __name__ == "__main__":
    test_topic_analysis()








