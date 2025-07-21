# Run with:
# python -m tests.test_pipeline

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import build_speaker_segments
from text_utils.analyzer import Analyzer
from dotenv import load_dotenv
from datetime import datetime
import os
import json

def test_full_pipeline():
    print("\t*** Running full pipeline test...")
    
    # Config
    sample_name = "sample_1_9m"
    sample_path = f"data/raw/{sample_name}.wav"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"data/processed/test_full_pipeline/{sample_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    transcriber = Transcriber(model_size="medium")
    diarizer = Diarizer()
    analyzer = Analyzer()

    # Transcribe
    transcribe_result = transcriber.transcribe(sample_path, word_timestamps=True)
    print(f"\t*** Transcribed {len(transcribe_result['segments'])} segments.")

    # Diarize
    diarization_result = diarizer.diarize_audio(sample_path)
    print(f"\t*** Diarization complete. Found {len(diarization_result['segments'])} segments")
    
    # Align words to speaker segments 
    speaker_segments = build_speaker_segments(transcribe_result["segments"], diarization_result["segments"])

    # Analyze and extract themes
    full_transcript = transcribe_result["complete_text"]
    theme_result = analyzer.extract_themes(full_transcript)
    themes = theme_result["json_themes"]
    print(f"\t*** Theme analysis complete. Extracted {len(themes)} total themes (including default).")

    # Classify each speaker segment
    classified_segments = analyzer.classify_all_segments(speaker_segments, themes) 
    print(f"\t*** Segment classification complete.")   

    # Save final output
    output_path = os.path.join(output_dir, f"final_classified_segments.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classified_segments, f, indent=2, ensure_ascii=False)
    
    print(f"\t***✅ Full pipeline completed successfully.")
    print(f"\t*** Output written to: {output_path}")


if __name__ == "__main__":
    load_dotenv()
    test_full_pipeline()
