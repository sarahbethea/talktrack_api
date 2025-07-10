from pyannote.audio import Pipeline
from dotenv import load_dotenv
import time
import torch
import os

# Load from .env to access Huggingface access token
load_dotenv()
token = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Diarizer:
    def __init__(self, model_name="pyannote/speaker-diarization-3.1", hf_token=token):
        print(f"\t*** Loading diarization model: {model_name} ***")
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        print("\t*** Model loaded successfully ***")

        # Move to GPU if available
        if device == "cuda":
            model = model.to(torch.device(device))


    def diarize_audio(self, file_path):
        """
        Diarizes audio using pyannote.audio
        
        Args:
            model (pyannote.audio): pretrained diarization model. 
            file_path (str): Path to audio file.

        Returns:

        """
        print("\t*** Running inference with pyannote.audio ***")

    
        # Run inference
        start_time = time.time()
        result = self.pipeline(file_path)
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
        cleaned_segments = self._clean_diarization_output(raw_segments)

        return {
            "segments": cleaned_segments,
            "raw_segments": raw_segments,
            "inference_time": round(inference_time, 2)
        }
    

    def _clean_diarization_output(self, segments, min_duration=3.0, max_gap=5.0, format_timestamps=True):
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
        print("\t*** Cleaning up diarization output ***")

        if not segments:
            return segments
        
        # Step 1: Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Step 2: Initial merge
        merged_segments = self._merge_consecutive_speakers(sorted_segments, max_gap)
        
        # Step 3: Filter short segments
        filtered_segments = self._filter_short_segments(merged_segments, min_duration)
        
        # Step 4: Merge again (in case filtering created new consecutive same-speaker segments)
        final_segments = self._merge_consecutive_speakers(filtered_segments, max_gap)
        
        # Step 5: Format timestamps if requested
        if format_timestamps:
            for segment in final_segments:
                segment["start_formatted"] = self._seconds_to_mmss(segment["start"])
                segment["end_formatted"] = self._seconds_to_mmss(segment["end"])
        
        return final_segments
    

    def _merge_consecutive_speakers(self, segments, max_gap=5.0):
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
    
    def _filter_short_segments(self, segments, min_duration=3.0):
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


    def _seconds_to_mmss(self, seconds):
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


        