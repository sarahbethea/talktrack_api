# Core pipeline

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import build_speaker_transcript, build_speaker_segments, generate_json_segments
from text_utils.analyzer import Analyzer
from utils.progress import update_progress 
from dotenv import load_dotenv
from datetime import datetime
import os, time, json
import logging


def run_pipeline(audio_path: str, job_id: str = "unknown") -> dict:
    logging.info(f"[{job_id}] 🚀 Starting pipeline for {audio_path}")
    metrics = {}

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
        logging.info(f"[{job_id}] ✅ Audio transcription completed in {metrics['transcription']}s")

        # Diarize
        update_progress(job_id, "diarizing", 20)
        d_start = time.time()
        diarization_result = diarizer.diarize_audio(audio_path)
        metrics["diarization"] = round(time.time() - d_start, 2)
        logging.info(f"[{job_id}] ✅ Audio diarization completed in {metrics['diarization']}s")

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
        logging.info(f"[{job_id}] ✅ Theme extraction completed in {metrics['theme_extraction']}s")

        # Classify segments
        update_progress(job_id, "classifying_segments", 70)
        classify_start = time.time()
        classified_segments = analyzer.classify_all_segments(segments, themes)
        metrics["classification"] = round(time.time() - classify_start, 2)
        logging.info(f"[{job_id}] ✅ Classification completed in {metrics['classification']}s")

        metrics["total"] = round(time.time() - total_start, 2)
        logging.info(f"[{job_id}] 🏁 Pipeline completed in {metrics['total']}s")

        return {
            "segments": classified_segments,
            "themes": themes,
            # "transcript": transcript,
            "metrics": metrics
        }

    except Exception as e:
        logging.exception(f"[{job_id}] ❌ Pipeline failed: {str(e)}")
        raise