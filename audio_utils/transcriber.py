# Whisper transcription logic

import whisper 
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device) # Other options: "medium", "large"

print(f"Using device: {device}")

def transcribe_audio(file_path):
    """
    Transcribes audio using Whisper.
    
    Args:
        file_path (str): Path to audio file.
    
    Returns:
        dict: Contains full text and segment-level timestamps.
    """
    start_time = time.time()
    result = model.transcribe(file_path)
    end_time = time.time()
    inference_time = end_time - start_time

    return {
        "text": result["text"],
        "segments": result["segments"],
        "inference_time": round(inference_time, 2)
    }