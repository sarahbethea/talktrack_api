"""
Thin wrapper around faster-whisper for transcription.

Responsibilities:
- Load a Whisper model (CPU/GPU with appropriate compute_type).
- Transcribe audio and return segments + full text + inference time.
"""
from faster_whisper import WhisperModel
import torch
import time
import logging
from typing import Any
import os

logger = logging.getLogger(__name__)

class Transcriber:
    """
    Speech-to-text helper using faster-whisper.

    Args:
        model_size: Whisper model size string (e.g., "small", "medium", "large-v3").
                    Defaults to "medium".
    """
    def __init__(self, model_size="medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Common faster-whisper choices: float16 on GPU, int8 on CPU (fast, compact)
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model_size = model_size

        logger.info(
            "Loading Faster Whisper (size: %s, device: %s, compute_type: %s)", 
            model_size, self.device.upper(), self.compute_type
        )
        self.model = WhisperModel(
            model_size, 
            device=self.device, 
            compute_type=self.compute_type,
            download_root=os.getenv("WHISPER_CACHE", "/cache/whisper")
        )
        logger.info("Model loaded successfully")

    
    def transcribe(self, audio_path: str, word_timestamps: bool =True) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            word_timestamps: If True, include per-word timestamps when available.

        Returns:
            dict with:
                - "segments": list of segment dicts (start, end, text, optional words[])
                - "complete_text": concatenated transcript text
                - "inference_time": float seconds (rounded)
        """
        logger.info("Transcribing: %s (word_timestamps=%s)", audio_path, word_timestamps)
        start_time = time.time()
        segments, _ = self.model.transcribe(audio_path, word_timestamps=word_timestamps)
        end_time = time.time()  
        inference_time = end_time - start_time

        logger.info("Transcription complete. Inference time: %.2fs", inference_time)

        segment_list: list[dict[str, Any]] = []
        for segment in segments:
            segment_dict: dict[str, Any] = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text
            }

            if hasattr(segment, "words") and segment.words is not None:
                segment_dict["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    } 
                    for w in segment.words
                ]
            
            segment_list.append(segment_dict)

        # Remove timestamps and merge segments to generate complete text. 
        complete_text = " ".join([segment["text"].strip() for segment in segment_list])

        return {
            "segments": segment_list,
            "complete_text": complete_text,
            "inference_time": round(inference_time, 2)
        }