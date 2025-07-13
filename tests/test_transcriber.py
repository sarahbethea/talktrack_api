# Run with command:
# python -m tests.test_transcriber

from audio_utils.transcriber import Transcriber
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import os

# Config
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"
output_path = f"data/processed/test_transcriber/test_transcriber_output_{sample}_{timestamp}.json"

# Load environment variables (not required for Whisper, but good for consistency)
load_dotenv()

def test_transcription():
    print("\t*** Starting Transcription Test")

    # Load model
    transcriber = Transcriber(model_size="medium")  # Other options: "medium" or "base"

    # Run transcription
    start_time = time.time()
    result = transcriber.transcribe(audio_path, word_timestamps=True)
    end_time = time.time()

    print(f"\t*** Transcription completed in {round(end_time - start_time, 2)}s")
    print(f"\t*** Found {len(result['segments'])} transcript segments")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\t*** Output written to {output_path}")

if __name__ == "__main__":
    test_transcription()