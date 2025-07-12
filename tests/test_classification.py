# Run with:
# python -m tests.test_classification

from text_utils.analyzer import TopicAnalyzer
from dotenv import load_dotenv
import os
import json

sample = "sample_1_9m"
themes_path = "data/processed/test_topic_analysis/generated_themes_sample_1_9m.json"
segments_path = "data/processed/test_audio_processing/test_json_transcript_sample_1_9m.json"
output_path = f"data/processed/test_classification/classified_segments_{sample}.json"

load_dotenv()

def test_classification():
    # Load themes from JSON
    try:
        with open(themes_path, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except Exception as e:
        print(f"ERROR could not load themes: {e}")
        return

    # Load segments from JSON
    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        print(f"ERROR could not load segments: {e}")
    
    # Initialize LLama model
    print("\t*** Loading TopicAnalyzer...")
    analyzer = TopicAnalyzer()

    print("\t*** Classifying each segment")

    # # Classify each segment
    # for i, segment in enumerate(segments):
    #     result = analyzer.classify_segment(segment["text"], themes)
    #     segment["theme_title"] = result.get("title_theme", "Uncategorized")
    #     segment["summary"] = result.get("summary", "")

    #     print(f"\t[{i+1}/{len(segments)}] Theme: {segment["theme_title"]}")

    segments = analyzer.classify_all_segments(segments, themes)
    
    # Save output 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    print(f"\t*** Classified segments saved to {output_path}")


if __name__ == "__main__":
    test_classification()


