# Speaker diarization logic
import time

def diarize_audio(model, file_path):
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

    for turn, _, speaker in result.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })


    return {
        "segments": segments,
        "inference_time": round(inference_time, 2)
    }

    



if __name__ == "__main__":
    pass