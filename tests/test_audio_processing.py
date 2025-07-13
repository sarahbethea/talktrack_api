# Run with command
# python -m tests.test_audio_processing

from pipeline.audio_processing import process_audio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from dotenv import load_dotenv
import torch
import os
import json
import time

# Config
sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"
json_output_path = f"data/processed/test_audio_processing/test_json_transcript_{sample}.json"
transcript_output_path = f"data/processed/test_audio_processing/complete_transcript_{sample}.txt"

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")

def test_audio_processing():
    # Load models
    transcriber = Transcriber(model_size="small") 
    diarizer = Diarizer(hf_token=token)

    # Process audio
    start_time = time.time()
    result = process_audio(transcriber, diarizer, audio_path)
    end_time = time.time()
    print(f"\t*** Audio processing completed in {end_time - start_time}s")

    json_transcript = result["json_transcript"]
    complete_transcript = result["complete_transcript"]

    # Save json output
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_transcript, f, indent=2, ensure_ascii=False)

    print(f"\t*** JSON transcript saved to {json_output_path}")

    # Save complete transcript for LLM 
    os.makedirs(os.path.dirname(transcript_output_path), exist_ok=True)
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        f.write(complete_transcript)

    print(f"\t*** Complete transcript saved to {transcript_output_path}")


if __name__ == "__main__":
    test_audio_processing()