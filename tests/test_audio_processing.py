# Run with command
# python -m tests.test_audio_processing

from pipeline.audio_processing import process_audio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torch
import os
import time

# Config
sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"
output_path = f"data/processed/test_audio_processing/test_complete_transcript_{sample}.txt"

# Load environment variables
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"Using device: {device}, compute_type: {compute_type}")

# Load models
transcription_model = WhisperModel("small", compute_type=compute_type, device=device)
diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
if device == "cuda":
    model = diarization_model.to(torch.device(device))


# Process audio
start_time = time.time()
complete_transcript = process_audio(transcription_model, diarization_model, audio_path)
end_time = time.time()
print(f"\t*** Audio processing completed in {end_time - start_time}s ***")

# Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("COMPLETE TRANSCRIPT\n\n")
    f.write(complete_transcript)

print(f"\t*** Transcript saved to {output_path} ***")