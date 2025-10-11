"""
Core end-to-end pipeline for TalkTrack.

Steps:
1) Load models (Transcriber, Diarizer, Analyzer)
2) Transcribe audio
3) Diarize speakers
4) Align outputs into segments & transcript
5) Extract themes
6) Classify segments by theme
7) Return structured results + basic timing metrics
"""

from typing import Any
from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import (
    build_speaker_transcript, 
    build_speaker_segments, 
    generate_json_segments,
)
from text_utils.analyzer import Analyzer
from utils.progress import update_progress 
import time
import logging

logger = logging.getLogger(__name__)

def run_pipeline(audio_path: str, job_id: str = "unknown") -> dict[str, Any]:
    """Run the full analysis pipeline and return segments, themes, and metrics."""
    logger.info("[%s] 🚀 Starting pipeline for %s", job_id, audio_path)
    metrics: dict[str, float] = {}

    try:
        total_start = time.time()

        # Load models
        update_progress(job_id, "loading_models", 5)
        transcriber = Transcriber()
        diarizer = Diarizer()
        analyzer = Analyzer()

        # Transcribe
        update_progress(job_id, "transcribing", 15)
        t_start = time.time()
        transcription_result = transcriber.transcribe(audio_path, word_timestamps=True)
        metrics["transcription"] = round(time.time() - t_start, 2)
        logger.info("[%s] ✅ Audio transcription completed in %.2fs", job_id, metrics["transcription"])

        # Diarize
        update_progress(job_id, "diarizing", 20)
        d_start = time.time()
        diarization_result = diarizer.diarize_audio(audio_path)
        metrics["diarization"] = round(time.time() - d_start, 2)
        logger.info("[%s] ✅ Audio diarization completed in %.2fs", job_id, metrics["diarization"])

        # Align outputs 
        update_progress(job_id, "segmenting", 30)
        speaker_segments = build_speaker_segments(
            transcription_result["segments"],
            diarization_result["segments"]  
        )

        segments = generate_json_segments(speaker_segments)
        transcript = build_speaker_transcript(speaker_segments)

        # Extract themes
        update_progress(job_id, "extracting_themes", 50)
        theme_start = time.time()
        theme_result = analyzer.extract_themes(transcript)
        themes = theme_result["json_themes"]
        metrics["theme_extraction"] = round(time.time() - theme_start, 2)
        logger.info("[%s] ✅ Theme extraction completed in %.2fs", job_id, metrics["theme_extraction"])
        
        # Classify segments
        update_progress(job_id, "classifying_segments", 70)
        classify_start = time.time()
        classified_segments = analyzer.classify_all_segments(segments, themes)
        metrics["classification"] = round(time.time() - classify_start, 2)
        logger.info("[%s] ✅ Classification completed in %.2fs", job_id, metrics["classification"])

        metrics["total"] = round(time.time() - total_start, 2)
        logger.info("[%s] 🏁 Pipeline completed in %.2fs", job_id, metrics["total"])

        return {
            "segments": classified_segments,
            "themes": themes,
            # "transcript": transcript,
            "metrics": metrics
        }

    except Exception as e:
        logger.exception(f"[{job_id}] ❌ Pipeline failed: {str(e)}")
        raise