"""
Utilities to align transcription (words/timestamps) with diarized speaker segments,
and to produce both human-readable and JSON-ready outputs.
"""

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from utils.progress import update_progress 
import logging
from typing import Any

logger = logging.getLogger(__name__)

def process_audio(transcriber: Transcriber, diarizer: Diarizer, audio_path: str, job_id: str) -> dict[str, Any]:
    """
    Run transcription and speaker diarization on an audio file, and format results
    for downstream processing (e.g., summarization, annotation, Premiere integration).

    This function:
    1. Transcribes the audio using a preloaded FasterWhisper model.
    2. Performs speaker diarization using a preloaded pyannote model.
    3. Aligns transcript segments with diarized speaker segments.
    4. Creates both a clean human-readable transcript and a JSON-formatted list of segments.

    Args:
        transcription_model (Transcriber): An initialized Transcriber instance.
        diarization_model (Diarizer): An initialized Diarizer instance.
        audio_path (str): Path to the raw audio file to process.

    Returns:
        dict: Contains two keys:
            - "json_transcript": list of enriched speaker segments (for classification, markers)
            - "complete_transcript": a plain-text transcript formatted by speaker and timestamp
    """
    logger.info("[%s] Processing audio", job_id)

    # Transcribe
    update_progress(job_id, "transcribing", 5)
    transcription_result = transcriber.transcribe(audio_path, word_timestamps=True)

    # Diarize
    update_progress(job_id, "diarizing", 10)
    diarization_result = diarizer.diarize_audio(audio_path)

    # Combine them
    update_progress(job_id, "segmenting", 30)
    speaker_segments = build_speaker_segments(
        transcription_result["segments"],
        diarization_result["segments"],  # These are your cleaned segments
    )

    # Create complete transcript
    complete_transcript = build_speaker_transcript(speaker_segments)

    # Create JSON formatted transcript
    json_segments = generate_json_segments(speaker_segments)

    return {
        "json_transcript": json_segments,
        "complete_transcript": complete_transcript
    }

def build_speaker_segments(transcription_segments: list[dict], diarization_segments: list[dict]) -> list[dict]:
    """
    Assign individual words (with timestamps) to the correct speaker segment,
    then build clean speaker-labeled blocks of text.

    Args:
        transcription_segments: list of segments from FasterWhisper with "words"
        diarization_segments: list of diarized speaker segments

    Returns:
        list of dicts with speaker, start/end, text, and duration
    """
    logger.info("Building speaker segments from word-level timestamps")

    # Flatten all words from transcriber output into single list
    all_words = []
    for segment in transcription_segments:
        if "words" in segment:
            all_words.extend(segment["words"])
    
     # Assign each word to the best diarized speaker segment (based on overlap)
    segment_word_map = {i: [] for i in range(len(diarization_segments))} # Create empty list for each diarized speaker segment

    # Loop through words and match them with diarization segment
    for word in all_words:
        best_match = None
        best_overlap = 0.0

        for i, d in enumerate(diarization_segments):
            overlap = compute_overlap(word["start"], word["end"], d["start"], d["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = i

        if best_match is not None:
            segment_word_map[best_match].append(word)

    # Build the final speaker segments
    speaker_segments = []
    for i, d in enumerate(diarization_segments):
        # Join words that matched with each segment
        words = segment_word_map[i]
        full_text = " ".join(w["word"] for w in words)

        segment = {
            "segment_id": i,
            "start": d["start"],
            "end": d["end"],
            "start_formatted": d.get("start_formatted"),
            "end_formatted": d.get("end_formatted"),
            "speaker": d["speaker"],
            "text": full_text,
            "duration": round(d["end"] - d["start"], 2)
        }

        speaker_segments.append(segment)

    return speaker_segments


def extract_text_for_speaker_segments(transcription_segments: list[dict], diarization_segments: list[dict]) -> list[dict]:
    """
    For each cleaned diarization segment, extract all transcribed text 
    that was spoken during that time period.
    
    Args:
        transcription_segments: List of small transcript segments
        diarization_segments: List of cleaned, larger speaker segments
    
    Returns:
        List of segments with speaker, timestamps, and complete text
    """
    logger.info("Extracting text for speaker segments")

    # Track which transcript segments go with which diarization segment
    segment_assignments = {i: [] for i in range(len(diarization_segments))}

    for t in transcription_segments:
        best_match_index = None
        best_overlap = 0.0

        for i, d in enumerate(diarization_segments):
            overlap = compute_overlap(
                t["start"], t["end"],
                d["start"], d["end"]
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_match_index = i
        
        if best_match_index is not None:
            segment_assignments[best_match_index].append(t["text"].strip())
    
    # Build final speaker segments
    speaker_segments_with_text = []
    for i, diar_segment in enumerate(diarization_segments):
        full_text = " ".join(segment_assignments[i])

        enhanced_segment = {
            "start": diar_segment["start"],
            "end": diar_segment["end"],
            "speaker": diar_segment["speaker"],
            "text": full_text,
            "duration": round(diar_segment["end"] - diar_segment["start"], 2)
        }

        if "start_formatted" in diar_segment:
            enhanced_segment["start_formatted"] = diar_segment["start_formatted"]
            enhanced_segment["end_formatted"] = diar_segment["end_formatted"]

        speaker_segments_with_text.append(enhanced_segment)

    return speaker_segments_with_text


def compute_overlap(start1, end1, start2, end2):
    """ Return the amount of temporal overlap between two intervals. """
    overlap_start = max(start1, start2) # take later of two start times
    overlap_end = min(end1, end2) # take earlier of two end times 
    return max(0.0, overlap_end - overlap_start) # Return difference between start and end, use max to avoid negative values 


def build_speaker_transcript(speaker_segments_with_text: list[dict]) -> str:
    """
    Generate a full, readable transcript from speaker-segmented text data.

    Each segment is formatted with a speaker label, start and end timestamps, and the spoken text.
    Only non-empty text segments are included. The final output is a single string where each 
    segment is separated by two newlines, making it suitable for display or LLM input.

    Args:
        speaker_segments_with_text (list[dict]): A list of segments with keys:
            - "speaker" (str): The speaker label
            - "start_formatted" (str): Timestamp in MM:SS.ss or HH:MM:SS.ss
            - "end_formatted" (str): Timestamp in MM:SS.ss or HH:MM:SS.ss
            - "text" (str): The transcribed text

    Returns:
        str: A human-readable multi-line transcript, labeled and timestamped per segment.
    """
    logger.info("Creating complete speaker transcript")

    transcript_parts: list[str] = []

    for segment in speaker_segments_with_text:
        speaker_label = segment["speaker"]
        start_time = segment["start_formatted"]
        end_time = segment["end_formatted"]
        text = segment["text"]
        
        if text.strip():  # Only add if there's actual text
            transcript_parts.append(f"[{speaker_label}][{start_time} - {end_time}]: {text}")
    
    return "\n\n".join(transcript_parts)


def generate_json_segments(speaker_segments_with_text: list[dict]) -> list[dict]:
    """
    Generate structured JSON segments output from enhanced speaker segments.

    Args:
        speaker_segments_with_text (list[dict]): List of dicts with 'start', 'end', 'text', etc.

    Returns:
        List[Dict]: Structured list of segments ready for classification/annotation.
    """
    logger.info("Generating structured JSON segments")

    json_segments: list[dict] = []

    for i, segment in enumerate(speaker_segments_with_text):
        seg = {
            "segment_id": i,
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "start_formatted": segment["start_formatted"],
            "end_formatted": segment["end_formatted"],
            "speaker": segment["speaker"],
            "text": segment["text"],
            "summary": None,
            "theme_title": None,
            "theme_id": None
        }
        json_segments.append(seg)
    
    return json_segments
