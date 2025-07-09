import time

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model("small", device=device) # Other options: "medium", "large"

# print(f"Using device: {device}")

def transcribe_with_whisper(model, file_path):
    """
    Transcribes audio using Whisper.
    
    Args:
        model (WhisperModel): pretrained model.
        file_path (str): Path to audio file.
    
    Returns:
        dict: Contains full text and segment-level timestamps.
    """
    start_time = time.time()
    result = model.transcribe(file_path) #run inference
    end_time = time.time()
    inference_time = end_time - start_time

    return {
        "text": result["text"],
        "segments": result["segments"],
        "inference_time": round(inference_time, 2)
    }


def transcribe_with_faster_whisper(model, audio_path, word_timestamps=False):
    """
    Transcribes audio using a provided faster-whisper model.

    Args:
        model (WhisperModel): A preloaded faster-whisper model.
        file_path (str): Path to audio file.
        word_timestamps (bool): Whether to include word-level timing.

    Returns:
        dict: Contains list of segments, total inference time, and the complete text without timestamps. 
    """
    print("\t*** Running inference with Faster Whisper ***")

    start_time = time.time()
    # Run inference
    segments, _ = model.transcribe(audio_path, word_timestamps=word_timestamps)
    end_time = time.time()
    inference_time = round(end_time - start_time, 2)

    print(f"\t*** Transcription complete. Inference time: {inference_time} ***")

    segment_list = []
    for segment in segments:
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    # Remove timestampts and merge segments to generate complete text. 
    complete_text = " ".join([segment["text"].strip() for segment in segment_list])

    return {
        "segments": segment_list,
        "inference_time": round(inference_time, 2),
        "complete_text": complete_text
    }




