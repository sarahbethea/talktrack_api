# Run with command:
# python -m tests.test_diarizer

from audio_utils.diarizer import Diarizer
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import os

# Config
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"
output_path = f"data/processed/test_diarization/test_diarizer_output_{sample}_{timestamp}.json"

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN") 


def test_diarization():
    """
    Test pyannote.audio
    """
    # Load model
    diarizer = Diarizer(hf_token=token)

    print(f"\t*** Testing pyannote.audio. Using device {diarizer.device.upper()}")

    # Run diarization
    start_time = time.time()
    result = diarizer.diarize_audio(audio_path)
    end_time = time.time()

    print(f"\t*** Diarization completed in {round(end_time - start_time, 2)}s")
    print(f"\t*** Found {len(result['segments'])} cleaned segments from {len(result['raw_segments'])} raw segments")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)

    print(f"\t*** Output written to {output_path}")


if __name__ == "__main__":
    test_diarization()