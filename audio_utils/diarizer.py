"""
Thin wrapper around pyannote.audio for speaker diarization.

Responsibilities:
- Load the pyannote pipeline (optionally on GPU).
- Run diarization and return raw + cleaned segments.
- Provide small utilities to merge/filter segments and format timestamps.
"""

from typing import Any, Optional
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from huggingface_hub import login
import torchaudio
import tempfile
import time
import torch
import os
import logging

# Load from .env to access Huggingface access token
load_dotenv()
token = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)

class Diarizer:
    """
    Speaker diarization helper.

    Args:
        model_name: Hugging Face model id for pyannote diarization.
        hf_token:   Hugging Face access token (falls back to HF_TOKEN env).
    """
    def __init__(self, model_name: str ="pyannote/speaker-diarization-3.1", hf_token: Optional[str] = None):
        hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
        )

        if not hf_token:
            # You *must* pass a token for gated pyannote models in a fresh container
            # (no cached login). Raise a clear error now rather than a 403 later.
            raise RuntimeError(
                "Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment."
            )
        
        try:
            login(token=hf_token)  # no other kwargs
        except Exception as e:
            logger.warning("HF login failed (continuing because we pass token directly): %s", e)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading diarization model: %s (device: %s)", model_name, self.device)
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        logger.info("Model loaded successfully")

        # Move to GPU if available
        if self.device == "cuda":
            self.pipeline = self.pipeline.to(torch.device(self.device))


    def diarize_audio(self, file_path: str) -> dict:
        """
        Run diarization on an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            dict with:
                - "segments": cleaned list[dict] of speaker segments
                - "raw_segments": raw list[dict] from pyannote
                - "inference_time": float seconds (rounded)
        """
        logger.info("Running diarization inference with pyannote.audio on %s", file_path)

        # Normalize input to WAV mono 16kHz (pyannote requirement)
        norm_path = self._to_wav_mono16k(file_path)

        # Run inference
        try:
            start_time = time.time()
            result = self.pipeline(norm_path)
            end_time = time.time()
            inference_time = end_time - start_time
        finally:
            try:
                os.remove(norm_path)
            except OSError:
                pass

        logger.info("Diarization complete in %.2fs", inference_time)

        # Extract raw segments
        raw_segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            raw_segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })

        # Clean up segments
        cleaned_segments = self._clean_diarization_output(raw_segments)

        return {
            "segments": cleaned_segments,
            "raw_segments": raw_segments,
            "inference_time": round(inference_time, 2)
        }
    
    def _to_wav_mono16k(self, in_path: str) -> str:
        """Return a temp WAV file (mono, 16 kHz) path for a given audio file."""
        wav_fd, out_path = tempfile.mkstemp(prefix="tt_diar_", suffix=".wav")
        os.close(wav_fd)  # we'll write it with torchaudio.save

        # Load (let torchaudio handle mp3/m4a/wav)
        wav, sr = torchaudio.load(in_path)  # shape: [channels, num_samples]

        # Mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # [1, N]

        # Resample -> 16k if needed
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)

        # Save as PCM WAV
        torchaudio.save(out_path, wav, sample_rate=target_sr, encoding="PCM_S", bits_per_sample=16)
        return out_path
    

    def _move_pipeline_to_cuda(self):
        """ (Unused helper) Move internal models inside the pyannote pipeline to GPU. """
        for name, model in self.pipeline.model.items():
            self.pipeline.model[name] = model.to(torch.device("cuda"))
        logger.debug("Pipeline models moved to CUDA")


    def _clean_diarization_output(
            self, 
            segments: list[dict[str, Any]], 
            min_duration: float = 3.0, 
            max_gap: float = 5.0, 
            format_timestamps: bool = True
    ) -> list[dict[str, Any]]:
        """
        Clean up diarization segments with multiple passes of merging and filtering.
        
        Args:
            segments (list): Raw diarization segments
            min_duration (float): Minimum segment duration to keep (seconds)
            max_gap (float): Maximum gap between segments to merge (seconds)
            format_timestamps (bool): Whether to format timestamps as MM:SS
        
        Returns:
            list[dict]: Fully cleaned segments
        """
        logger.debug("Cleaning diarization output: %d raw segments", len(segments) if segments else 0)

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
    

    def _merge_consecutive_speakers(
            self, segments: list[dict[str, Any]], max_gap: float = 5.0
    ) -> list[dict[str, Any]]:
        """
        Merge consecutive segments from the same speaker when the temporal gap is small.

        Args:
            segments: Segments sorted by start time.
            max_gap: Merge when gap between same-speaker segments ≤ max_gap seconds.

        Returns:
            Segments with consecutive same-speaker regions merged.
        """
        if not segments:
            return segments
        
        merged_segments: list[dict[str, Any]] = []
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
    
    
    def _filter_short_segments(
            self, segments: list[dict[str, Any]], min_duration: float = 3.0
    ) -> list[dict[str, Any]]:
        """
        Remove segments shorter than the minimum duration.

        Args:
            segments: Input segments.
            min_duration: Minimum duration to keep (seconds).

        Returns:
            Segments with short ones removed.
        """
        filtered_segments: list[dict[str, Any]] = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration >= min_duration:
                filtered_segments.append(segment)
            else:
                logger.debug("Filtered out short segment: %.2fs (%s)", duration, segment.get("speaker"))
        
        return filtered_segments


    def _seconds_to_mmss(self, seconds: float) -> str:
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


        