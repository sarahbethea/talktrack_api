from audio_utils.transcribe import transcribe_with_faster_whisper
from audio_utils.diarization import diarize_audio


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
        text = segment["text"]
        
        if text.strip():  # Only add if there's actual text
            transcript_parts.append(f"[{speaker_label}]: {text}")
    
    return "\n\n".join(transcript_parts)

def process_audio(transcription_model, diarization_model, audio_path):
    print("\t*** Processing audio ***")

    # Get results 
    transcription_result = transcribe_with_faster_whisper(transcription_model, audio_path)
    diarization_result = diarize_audio(diarization_model, audio_path)

    # Combine them
    speaker_segments = extract_text_for_speaker_segments(
        transcription_result["segments"],
        diarization_result["segments"]  # These are your cleaned segments
    )

    # Create clean transcript for LLM
    complete_speaker_transcript = create_complete_transcript_by_speaker(speaker_segments)

    return complete_speaker_transcript