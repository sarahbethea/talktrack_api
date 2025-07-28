from text_utils.analyzer import Analyzer
import os
import json
from datetime import datetime


def test_topic_analysis():
    sample_name = "sample_1_9m"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    transcript_path = "data/processed/test_audio_processing/audio_processing_transcript.txt"
    parsed_output_path = f"data/processed/test_topic_analysis/generated_themes_{sample_name}_{timestamp}.txt"
    json_output_path = f"data/processed/test_topic_analysis/generated_themes_{sample_name}_{timestamp}.json"
        
    # Read the transcript file
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        print(f"Transcript loaded: {len(transcript)} characters")
    except FileNotFoundError:
        print(f"Error: Could not find file {transcript_path}")
        return

    # Analyze topics
    print("\t*** Initializing topic analyzer...")
    model = Analyzer()

    print("\t*** Extracting themes...")
    result = model.extract_themes(transcript)

    print(f"\t*** Found {len(result['parsed_themes'])} themes")

    # Write results to output_path
    with open(parsed_output_path, "w", encoding="utf-8") as f:
        f.write("TOPIC ANALYSIS RESULTS:\n\n")

        f.write("PARSED THEMES:\n")
        f.write("-" * 20 + "\n")

        for i, theme in enumerate(result["parsed_themes"], 1):
            f.write(f"\n{i}. {theme.get('title', 'Unknown Title')}\n")
            f.write(f"   Description: {theme.get('description', 'No description')}\n")
            f.write(f"   Keywords: {', '.join(theme.get('keywords', []))}\n")


    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result["json_themes"], f, indent=2)

    print(f"\t*** Results written to: {json_output_path}")

if __name__ == "__main__":
    test_topic_analysis()








