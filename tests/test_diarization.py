# run with command: 
# python -m tests.test_diarization

from pyannote.audio import Pipeline
from audio_utils.diarization import diarize_audio, seconds_to_mmss
from dotenv import load_dotenv # Loads variables into the environment for access by os module 
import torch
import os

sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load from .env to access Huggingface access token
load_dotenv()
token = os.getenv("HF_TOKEN")

def test_diarization():
    """
    Test pyannote.audio
    """
    print(f"*** Testing pyannote.audio. Using device {device}")

    # Load model
    model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )

    # Move to GPU if available
    if device == "cuda":
        model = model.to(torch.device(device))


    # Run inference
    result = diarize_audio(model, audio_path)

    print(f"Inference completed in {result['inference_time']} seconds")
    print(f"Raw segments: {len(result['raw_segments'])}")
    print(f"Cleaned segments: {len(result['segments'])}")

    # Define output path for results 
    output_path = f"data/processed/test_diarization_{sample}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"SPEAKER DIARIZATION RESULTS:\n")
        f.write(f"Segments: {len(result['segments'])} (filtered from {len(result['raw_segments'])})\n\n")
        
        for segment in result["segments"]:
            duration_seconds = segment["end"] - segment["start"]
            duration_formatted = seconds_to_mmss(duration_seconds)
        
            if "start_formatted" in segment:
                # Use formatted timestamps
                f.write(f"{segment['start_formatted']} - {segment['end_formatted']} ({duration_formatted}): {segment['speaker']}\n")
            else:
                # Fallback to seconds
                f.write(f"{segment['start']}s - {segment['end']}s ({duration_seconds:.2f}s): {segment['speaker']}\n")
        
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    test_diarization()