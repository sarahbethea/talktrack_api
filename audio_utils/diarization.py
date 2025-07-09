# Speaker diarization logic
import time

def diarization(model, file_path):
    """
    Diarizes audio using pyannote.audio\
    
    Args:
        model (pyannote.audio): pretrained diarization model. 
        file_path (str): Path to audio file.

    Returns:

    """
    start_time = time.time()
    result = model(file_path)
    end_time = time.time()
    inference_time = end_time - start_time

    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


    return {
        "segments": segments,
        "inference_time": inference_time
    }

    



if __name__ == "__main__":
    diarization()