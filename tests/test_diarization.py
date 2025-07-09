# run with command: 
# python -m tests.test_diarization

from pyannote.audio import Pipeline
from audio_utils.diarization import diarization
from dotenv import load_dotenv # Loads variables into the environment for access by os module 
import torch
import os

sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load from .env to access Huggingface access token
load_dotenv()

def test_diarization():
    """
    Test pyannote.audio
    """
    print(f"*** Testing pyannote.audio. Using device {device}")

    # Load model
    model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )

    # Run inference
    result = diarization(model, audio_path)

    print(f"Transcription completed in {result['inference_time']} seconds")

    # Define output path for results 
    output_path = f"data/processed/test_diarization_{sample}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["segments"])
    
    print(f"Transcription written to {output_path}")


if __name__ == "__main__":
    test_diarization()