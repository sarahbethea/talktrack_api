# Run with:
#   python -m tests.test_pipeline
# or:
#   pytest -q tests/test_pipeline.py

"""
End-to-end pipeline test.

- Loads Transcriber, Diarizer, and Analyzer
- Runs transcription → diarization → alignment → theme extraction → classification
- Saves outputs + simple metrics into a timestamped folder under data/processed/
"""
from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import build_speaker_segments, process_audio
from text_utils.analyzer import Analyzer
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging
import os
import time
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config ---
SAMPLE_NAME = "sample_4_19m"
ROOT = Path(__file__).resolve().parents[1]  # repo root
SAMPLE_PATH = ROOT / "data" / "raw" / f"{SAMPLE_NAME}.wav"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = ROOT / "data" / "processed" / "test_full_pipeline" / f"{SAMPLE_NAME}_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

def test_full_pipeline(save_intermediates=False):
    """Run the full pipeline and write results/metrics to OUTPUT_DIR."""
    # Initialize metrics dictionary
    metrics = {
        "sample": SAMPLE_NAME,
        "timestamp": TIMESTAMP,
        "stages": {},
        "total_seconds": 0.0
    }

    logger.info("Running full pipeline test...")
    start_time = time.time()

    # Load models
    transcriber = Transcriber(model_size="medium")
    diarizer = Diarizer()
    analyzer = Analyzer()

    # Transcribe
    stage_start = time.time()
    transcription_result = transcriber.transcribe(SAMPLE_PATH, word_timestamps=True)
    stage_end = time.time()
    logger.info("Transcribed %d segments.", len(transcription_result['segments']))
    metrics["stages"]["transcription"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "segments": len(transcription_result["segments"]),
        "model_size": transcriber.model_size
    }

    # Diarize
    stage_start = time.time()
    diarization_result = diarizer.diarize_audio(SAMPLE_PATH)
    stage_end = time.time()
    logger.info("Diarization complete. Found %d segments", len(diarization_result['segments']))
    metrics["stages"]["diarization"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "segments": len(diarization_result["segments"])
    }

    # Align words to speaker segments 
    speaker_segments = build_speaker_segments(transcription_result["segments"], diarization_result["segments"])

    # Analyze and extract themes
    full_transcript = transcription_result["complete_text"]
    stage_start = time.time()
    theme_result = analyzer.extract_themes(full_transcript)
    stage_end = time.time()
    themes = theme_result["json_themes"]
    logger.info("Theme analysis complete. Extracted %d total themes (including default).", len(themes))
    metrics["stages"]["theme_analysis"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "themes_found": len(themes)
    }

    # Classify each speaker segment
    stage_start = time.time()
    classified_segments = analyzer.classify_all_segments(speaker_segments, themes) 
    stage_end = time.time()
    logger.info("Segment classification complete.")
    metrics["stages"]["classification"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "segments_classified": len(classified_segments),
        "success_rate": analyzer.last_classification_stats.get("successful", 0) / max(1, len(classified_segments)),
        "num_failed_segments": len(analyzer.failed_segments),
        "failed_segments": [
            {
                "segment_id": s.get("segment_id"),
                "start": s.get("start"),
                "end": s.get("end"),
                "text_snippet": s.get("text", "")[:60]
             } for s in analyzer.failed_segments
        ]
    }

    end_time = time.time()
    total_seconds = round(end_time - start_time, 2)
    metrics["total_seconds"] = total_seconds

    # Optionally save output of intermediate steps 
    if save_intermediates:
        save_json(transcription_result, os.path.join(OUTPUT_DIR, "transcription_result.json"))
        save_json(diarization_result, os.path.join(OUTPUT_DIR, "diarization_result.json"))
        save_json(speaker_segments, os.path.join(OUTPUT_DIR, "speaker_segments.json"))
        save_json(themes, os.path.join(OUTPUT_DIR, "themes.json"))

    # Save final output and metrics
    save_json(classified_segments, os.path.join(OUTPUT_DIR, "final_classified_segments.json"))
    save_json(metrics, os.path.join(OUTPUT_DIR, "metrics.json"))

    logger.info("✅ Full pipeline completed successfully in %ds.", total_seconds)
    logger.info("Output written to: %s", OUTPUT_DIR)


def save_json(data, path):
    """Write JSON to disk with fsync to ensure flush to storage."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    test_full_pipeline(save_intermediates=True)
