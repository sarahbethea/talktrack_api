# Speaker diarization logic
import time

def diarize_audio(model, file_path):
    """
    Diarizes audio using pyannote.audio\
    
    Args:
        model (pyannote.audio): pretrained diarization model. 
        file_path (str): Path to audio file.

    Returns:

    """
    start_time = time.time()
    result = model(file_path)
    end_time = time.time()
    inference_time = end_time - start_time

    # Extract raw segments
    raw_segments = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        raw_segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

    # Clean up segments
    cleaned_segments = clean_diarization_output(raw_segments)

    return {
        "segments": cleaned_segments,
        "raw_segments": raw_segments,
        "inference_time": round(inference_time, 2)
    }


def clean_diarization_output(segments, min_duration=3.0, max_gap=5.0):
    """
    Clean up diarization segments by merging consecutive same-speaker segments
    and filtering out very short segments.
    
    Args:
        segments (list): Raw diarization segments
        min_duration (float): Minimum segment duration to keep (seconds).
            Segments shorter than 3 seconds get filtered out. 
        max_gap (float): Maximum gap between segments to merge (seconds). 
            Gaps of max_gap seconds or less between the same speaker get merged. 
    
    Returns:
        list: Cleaned segments
    """

    # If empty list, return to avoid errors
    if not segments:
        return segments
    
    # Ensure segments are sorted in order of start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])

    # Create new list of merged segments
    merged_segments = []

    # Get copy of first segment so we dont modify original 
    current_segment = sorted_segments[0].copy()

    # Loop through following segments
    for next_segment in sorted_segments[1:]:
        # Calculate gap
        gap = next_segment["start"] - current_segment["end"]
        # Set boolean if same speaker
        same_speaker = current_segment["speaker"] == next_segment["speaker"]

        # If same speaker and small gap, absorb next segment into current
        if same_speaker and gap <= max_gap:
            current_segment["end"] = next_segment["end"]
        # Else keep segment as is and append, then move on to evaluate next segment
        else:
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    # Add last segment 
    merged_segments.append(current_segment)

    #Filter out short segments
    filtered_segments = []
    for segment in merged_segments:
        duration = segment["end"] - segment["start"]
        if duration >= min_duration:
            filtered_segments.append(segment)
        else:
            print(f"Filtered out short segment: {duration:.2f}s - {segment['speaker']}")

    return filtered_segments


    



if __name__ == "__main__":
    pass