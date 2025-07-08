from audio_utils.transcriber import transcribe_with_whisper, transcribe_with_faster_whisper
import torch
import whisper
from faster_whisper import WhisperModel
import json
import os

# run with command: 
# python -m tests.test_transcribe

sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
    

def test_w_transcription():
    """
    Test Whisper transcription
    """
    print(f"*** Testing Whisper. Using device: {device}")
    
    # Load model
    model = whisper.load_model("small", device=device) # Other options: "medium", "large"

    # Run inference
    result = transcribe_with_whisper(model, audio_path) # Run inference

    print(f"Transcription completed in {result['inference_time']} seconds")

    # Define output path for results
    output_path = f"data/processed/test_transcript_{sample}_whisper.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FULL TRANSCRIPT:\n")
        f.write(result["text"])
        f.write("\n\nSEGMENT TRANSCRIPTS:\n\n")

        for segment in result["segments"]:
            start = round(segment["start"], 2)
            end = round(segment["end"], 2)
            text = segment["text"]

            f.write(f"{start}s - {end}s {text}\n")

    print(f"Transcription written to {output_path}")



def test_fw_transcription():
    """
    Test Faster Whisper transcription
    """
    print(f"*** Testing Faster Whisper. Using device: {device}, compute_type: {compute_type}")

    # Load model
    model = WhisperModel("small", compute_type=compute_type, device=device)
    
    # Run inference, result will be type dictionary
    result = transcribe_with_faster_whisper(model, audio_path)
    
    # Pull values out of result
    inference_time = result["inference_time"]
    segments = result["segments"]

    # Define output path for results
    output_path = f"data/processed/test_transcript_{sample}_faster_whisper.txt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)

    print(f"Transcription written to {output_path}")
    print(f"Inference time: {inference_time}")


if __name__ == "__main__":
    test_w_transcription()
    test_fw_transcription()