# Run with:
# python -m tests.test_pipeline

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import build_speaker_segments
from text_utils.analyzer import Analyzer
from dotenv import load_dotenv
from datetime import datetime
import os
import time
import json

# Config
SAMPLE_NAME = "sample_4_19m"
SAMPLE_PATH = f"data/raw/{SAMPLE_NAME}.wav"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"data/processed/test_full_pipeline/{SAMPLE_NAME}_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_full_pipeline(save_intermediates=False):
    # Initialize metrics dictionary
    metrics = {
        "sample": SAMPLE_NAME,
        "timestamp": TIMESTAMP,
        "stages": {},
        "total_seconds": 0.0
    }

    print("\t*** Running full pipeline test...")
    start_time = time.time()

    # Load models
    transcriber = Transcriber(model_size="medium")
    diarizer = Diarizer()
    analyzer = Analyzer()

    # Transcribe
    stage_start = time.time()
    transcription_result = transcriber.transcribe(SAMPLE_PATH, word_timestamps=True)
    stage_end = time.time()
    print(f"\t*** Transcribed {len(transcription_result['segments'])} segments.")
    metrics["stages"]["transcription"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "segments": len(transcription_result["segments"]),
        "model_size": transcriber.model_size
    }

    # Diarize
    stage_start = time.time()
    diarization_result = diarizer.diarize_audio(SAMPLE_PATH)
    stage_end = time.time()
    print(f"\t*** Diarization complete. Found {len(diarization_result['segments'])} segments")
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
    print(f"\t*** Theme analysis complete. Extracted {len(themes)} total themes (including default).")
    metrics["stages"]["theme_analysis"] = {
        "duration_seconds": round(stage_end - stage_start, 2),
        "themes_found": len(themes)
    }

    # Classify each speaker segment
    stage_start = time.time()
    classified_segments = analyzer.classify_all_segments(speaker_segments, themes) 
    stage_end = time.time()
    print(f"\t*** Segment classification complete.")   
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

    print(f"\t***✅ Full pipeline completed successfully in {total_seconds}s.")
    print(f"\t*** Output written to: {OUTPUT_DIR}")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    load_dotenv()
    test_full_pipeline(save_intermediates=True)
