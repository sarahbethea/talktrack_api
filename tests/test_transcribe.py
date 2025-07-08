from audio_utils.transcriber import transcribe_audio

# run with command: 
# python -m tests.test_transcribe

sample = "sample_1_9m"
audio_path = f"data/raw/{sample}.wav"
output_path = f"data/processed/test_transcript_{sample}.txt"

result = transcribe_audio(audio_path)

print(f"Transcription completed in {result['inference_time']} seconds")

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