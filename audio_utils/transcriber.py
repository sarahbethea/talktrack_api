from faster_whisper import WhisperModel
import torch
import time

class Transcriber:
    def __init__(self, model_size="medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"\t*** Loading Faster Whisper (size: {model_size}, device: {self.device.upper()}, compute_type: {self.compute_type})")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("\t*** Model loaded successfully")

    
    def transcribe(self, audio_path: str, word_timestamps: bool =True) -> dict:
        print(f"\t*** Transcribing: {audio_path}")
        start_time = time.time()
        segments, _ = self.model.transcribe(audio_path, word_timestamps=True)
        end_time = time.time()  
        inference_time = end_time - start_time

        print(f"\t*** Transcription complete. Inference time: {inference_time}")

        segment_list = []
        for segment in segments:
            segment_dict = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text
            }

            if hasattr(segment, "words") and segment.words is not None:
                segment_dict["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    } 
                    for w in segment.words
                ]
            
            segment_list.append(segment_dict)

        # Remove timestampts and merge segments to generate complete text. 
        complete_text = " ".join([segment["text"].strip() for segment in segment_list])

        return {
            "segments": segment_list,
            "complete_text": complete_text,
            "inference_time": round(inference_time, 2)
        }