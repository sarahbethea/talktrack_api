# Run with command
# python -m tests.test_audio_processing

from pipeline.audio_processing import process_audio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from dotenv import load_dotenv
from datetime import datetime
import torch
import os
import json
import time

# Config
sample_name = "sample_1_9m"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sample_path = f"data/raw/{sample_name}.wav"
output_dir = f"data/processed/test_audio_processing/{sample_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")

def test_audio_processing():
    # Load models
    transcriber = Transcriber(model_size="medium") 
    diarizer = Diarizer(hf_token=token)

    # Process audio
    start_time = time.time()
    result = process_audio(transcriber, diarizer, sample_path)
    end_time = time.time()
    print(f"\t*** Audio processing completed in {end_time - start_time}s")

    json_transcript = result["json_transcript"]
    complete_transcript = result["complete_transcript"]

    # Save json output
    json_output_path = os.path.join(output_dir, f"audio_processing_output.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_transcript, f, indent=2, ensure_ascii=False)

    print(f"\t*** JSON transcript saved to {json_output_path}")

    # Save complete transcript for LLM 
    transcript_output_path = os.path.join(output_dir, f"audio_processing_transcript.txt")
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        f.write(complete_transcript)

    print(f"\t*** Complete transcript saved to {transcript_output_path}")

def test_process_audio():
    # Test the process_audio function with a sample audio file
    transcriber = Transcriber(model_size="medium")
    diarizer = Diarizer(hf_token=token)
    result = process_audio(transcriber, diarizer, sample_path)

    assert result is not None
    assert "json_transcript" in result
    assert "complete_transcript" in result
    assert isinstance(result["json_transcript"], list)
    assert isinstance(result["complete_transcript"], str)   
    print("\t*** process_audio test passed!")

if __name__ == "__main__":
    test_audio_processing()
    test_process_audio()