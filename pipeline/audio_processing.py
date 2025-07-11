from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer

def process_audio(transcription_model, diarization_model, audio_path):
    """
    Run transcription and diarization and format outputs for downstream use. 

    """
    print("\t*** Processing audio ***")

    # Get transcription 
    transcriber = Transcriber()
    transcription_result = transcriber.transcribe(audio_path)

    # Get diarization
    diarizer = Diarizer()
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

def extract_text_for_speaker_segments(transcription_segments, diarization_segments):
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


def create_complete_transcript_by_speaker(speaker_segments_with_text):
    """
    Create a clean, speaker-labeled transcript from the enhanced segments.
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


def generate_json_segments(speaker_segments_with_text):
    """
    Generate structured JSON segments output from enhanced speaker segments.

    Args:
        speaker_segments_with_text (list): List of dicts with 'start', 'end', 'text', etc.

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


