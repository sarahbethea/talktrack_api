# Run with:
# python -m tests.test_classification

from text_utils.analyzer import Analyzer
from dotenv import load_dotenv
from datetime import datetime
import os
import json

sample_name = "sample_1_9m"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
themes_path = "data/processed/test_topic_analysis/generated_themes_sample_1_9m.json"
segments_path = "data/processed/test_audio_processing/audio_processing_output.json"
output_dir = f"data/processed/test_classification/{sample_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"classified_segments.json")


load_dotenv()

def test_classification(benchmark_batch_size=False):
    # Load themes from JSON
    try:
        with open(themes_path, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except Exception as e:
        print(f"\t*** [ERROR] could not load themes: {e}")
        return

    # Load segments from JSON
    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        print(f"\t*** [ERROR] could not load segments: {e}")
    
    # Initialize LLama model
    print("\t*** Loading Analyzer...")
    analyzer = Analyzer()

    if benchmark_batch_size:
        for bsize in [1, 4, 8, 16]:
            print(f"\n\t=== Benchmarking batch_size={bsize} ===")
            analyzer.classify_all_segments(segments, themes, batch_size=bsize) 
    else:
        print("\t*** Classifying each segment")

        segments = analyzer.classify_all_segments(segments, themes)
        
        # Save output 
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        print(f"\t*** Classified segments saved to {output_path}")
    
    # Save failed segments for review
    if analyzer.failed_segments:
        failed_output_path = os.path.join(output_dir, f"failed_segments.json")
        with open(failed_output_path, "w", encoding="utf-8") as f:
            json.dump(analyzer.failed_segments, f, indent=2, ensure_ascii=False)
        print(f"\t*** Saved {len(analyzer.failed_segments)} failed segments to failed_segments.json")


if __name__ == "__main__":
    test_classification(benchmark_batch_size=False)


