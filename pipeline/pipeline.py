# Core pipeline

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import process_audio
from text_utils.analyzer import Analyzer
from dotenv import load_dotenv
from datetime import datetime
import os, time, json


def run_pipeline(audio_path: str, job_id: str = "unknown") -> dict:
    print(f"[{job_id}] 🚀 Starting pipeline for {audio_path}")
    metrics = {}

    try:
        total_start = time.time()

        # Load models
        transcriber = Transcriber()
        diarizer = Diarizer()
        analyzer = Analyzer()

        # Process audio (transcription + diarization + alignment)
        stage_start = time.time()
        audio_result = process_audio(transcriber, diarizer, audio_path)
        metrics["audio_processing"] = round(time.time() - stage_start, 2)
        print(f"[{job_id}] ✅ Audio processing completed in {metrics['audio_processing']}s")

        segments = audio_result["json_transcript"]
        transcript = audio_result["complete_transcript"]

        # Extract themes
        theme_start = time.time()
        theme_result = analyzer.extract_themes(transcript)
        themes = theme_result["json_themes"]
        metrics["theme_extraction"] = round(time.time() - theme_start, 2)
        print(f"[{job_id}] ✅ Theme extraction completed in {metrics['theme_extraction']}s")

        # Classify segments
        classify_start = time.time()
        classified_segments = analyzer.classify_all_segments(segments, themes)
        metrics["classification"] = round(time.time() - classify_start, 2)
        print(f"[{job_id}] ✅ Classification completed in {metrics['classification']}s")

        metrics["total"] = round(time.time() - total_start, 2)
        print(f"[{job_id}] 🏁 Pipeline completed in {metrics['total']}s")

        return {
            "segments": classified_segments,
            "themes": themes,
            # "transcript": transcript,
            "metrics": metrics
        }

    except Exception as e:
        print(f"[{job_id}] ❌ Pipeline failed: {str(e)}")
        raise