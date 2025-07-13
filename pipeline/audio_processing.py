from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer

def process_audio(transcriber: Transcriber, diarizer: Diarizer, audio_path: str) -> dict:
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
    print("\t*** Processing audio ***")

    # Transcribe
    transcription_result = transcriber.transcribe(audio_path)

    # Diarize
    diarization_result = diarizer.diarize_audio(audio_path)

    # Combine them
    speaker_segments = extract_text_for_speaker_segments(
        transcription_result["segments"],
        diarization_result["segments"]  # These are your cleaned segments
    )

    # Create complete transcript
    complete_transcript = create_complete_transcript_by_speaker(speaker_segments)

    # Create JSON formatted transcript
    json_segments = generate_json_segments(speaker_segments)

    return {
        "json_transcript": json_segments,
        "complete_transcript": complete_transcript
    }

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
    print("\t*** Extracting text for speaker segments ***")

    speaker_segments_with_text = []
    
    for diar_segment in diarization_segments:
        # Find all transcript segments that overlap with this speaker segment
        overlapping_text = []
        
        for transcript_segment in transcription_segments:
            # Check if transcript segment overlaps with speaker segment
            if (transcript_segment["start"] < diar_segment["end"] and 
                transcript_segment["end"] > diar_segment["start"]):
                overlapping_text.append(transcript_segment["text"].strip())
        
        # Combine all text for this speaker segment
        complete_text = " ".join(overlapping_text)
        
        # Create enhanced segment
        enhanced_segment = {
            "start": diar_segment["start"],
            "end": diar_segment["end"],
            "speaker": diar_segment["speaker"],
            "text": complete_text,
            "duration": round(diar_segment["end"] - diar_segment["start"], 2)
        }
        
        # Add formatted timestamps if they exist
        if "start_formatted" in diar_segment:
            enhanced_segment["start_formatted"] = diar_segment["start_formatted"]
            enhanced_segment["end_formatted"] = diar_segment["end_formatted"]
        
        speaker_segments_with_text.append(enhanced_segment)
    
    return speaker_segments_with_text


def create_complete_transcript_by_speaker(speaker_segments_with_text: list[dict]) -> str:
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
    print("\t*** Creating complete speaker transcript ***")

    transcript_parts = []
    
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
    print("\t*** Generating structured JSON transcript ***")

    json_segments = []

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


