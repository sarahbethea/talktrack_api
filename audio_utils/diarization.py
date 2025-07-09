# Speaker diarization logic
import time

def diarize_audio(model, file_path):
    """
    Diarizes audio using pyannote.audio
    
    Args:
        model (pyannote.audio): pretrained diarization model. 
        file_path (str): Path to audio file.

    Returns:

    """
    print("\t*** Running inference with pyannote.audio ***")

    start_time = time.time()
    # Run inference
    result = model(file_path)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"\t*** Diarization complete. Inference time: {inference_time}s ***")

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


def merge_consecutive_speakers(segments, max_gap=5.0):
    """
    Merge consecutive segments from the same speaker.
    
    Args:
        segments (list): Segments sorted by start time
        max_gap (float): Maximum gap between segments to merge (seconds).
            Gaps of max_gap seconds or less between the same speaker get merged. 
    
    Returns:
        list: Segments with consecutive same-speaker segments merged
    """
    if not segments:
        return segments
    
    merged_segments = []
    current_segment = segments[0].copy()
    
    for next_segment in segments[1:]:
        gap = next_segment["start"] - current_segment["end"]
        same_speaker = current_segment["speaker"] == next_segment["speaker"]
        
        if same_speaker and gap <= max_gap:
            # Merge: extend current segment to absorb the next
            current_segment["end"] = next_segment["end"]
        else:
            # Save current segment and move onto next segment
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    # Append the last segment
    merged_segments.append(current_segment)

    return merged_segments


def filter_short_segments(segments, min_duration=3.0):
    """
    Remove segments shorter than minimum duration.
    
    Args:
        segments (list): Input segments
        min_duration (float): Minimum segment duration to keep (seconds).
            Segments shorter than min_duration get filtered out. 
    
    Returns:
        list: Segments with short ones filtered out
    """
    filtered_segments = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if duration >= min_duration:
            filtered_segments.append(segment)
        else:
            print(f"Filtered out short segment: {duration:.2f}s - {segment['speaker']}")
    
    return filtered_segments


def clean_diarization_output(segments, min_duration=3.0, max_gap=5.0, format_timestamps=True):
    """
    Clean up diarization segments with multiple passes of merging and filtering.
    
    Args:
        segments (list): Raw diarization segments
        min_duration (float): Minimum segment duration to keep (seconds)
        max_gap (float): Maximum gap between segments to merge (seconds)
        format_timestamps (bool): Whether to format timestamps as MM:SS
    
    Returns:
        list: Fully cleaned segments
    """
    print("\t\t*** Cleaning up diarization output ***")

    if not segments:
        return segments
    
    # Step 1: Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    
    # Step 2: Initial merge
    merged_segments = merge_consecutive_speakers(sorted_segments, max_gap)
    
    # Step 3: Filter short segments
    filtered_segments = filter_short_segments(merged_segments, min_duration)
    
    # Step 4: Merge again (in case filtering created new consecutive same-speaker segments)
    final_segments = merge_consecutive_speakers(filtered_segments, max_gap)
    
    # Step 5: Format timestamps if requested
    if format_timestamps:
        for segment in final_segments:
            segment["start_formatted"] = seconds_to_mmss(segment["start"])
            segment["end_formatted"] = seconds_to_mmss(segment["end"])
    
    return final_segments


def seconds_to_mmss(seconds):
    """
    Convert seconds to MM:SS format

    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Time in MM:SS format
    """

    minutes = int(seconds // 60)
    secs = seconds % 60

    return f"{minutes}:{secs:05.2f}" # 05.2f gives us MM:SS.ss format



    
