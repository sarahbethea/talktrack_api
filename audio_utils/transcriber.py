from faster_whisper import WhisperModel
import torch
import time

class Transcriber:
    def __init__(self, model_size="small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"\t*** Loading Faster Whisper (size: {model_size}, device: {self.device.upper()}, compute_type: {self.compute_type})")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("\t*** Model loaded successfully")

    
    def transcribe(self, audio_path: str, word_timestamps=False) -> dict:
        print(f"\t*** Transcribing: {audio_path}")
        start_time = time.time()
        segments, _ = self.model.transcribe(audio_path, word_timestamps=word_timestamps)
        end_time = time.time()  
        inference_time = end_time - start_time

        print(f"\t*** Transcription complete. Inference time: {inference_time}")

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
            "complete_text": complete_text,
            "inference_time": round(inference_time, 2)
        }