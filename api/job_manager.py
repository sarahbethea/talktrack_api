# Background job runner and state manager
from pipeline.pipeline import run_pipeline
import os
import json

JOB_STATUS = {}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_pipeline_and_store(audio_path: str, job_id: str):
    try:
        JOB_STATUS[job_id] = "processing"
        results = run_pipeline(audio_path)
        save_results_to_disk(job_id, results)
        JOB_STATUS[job_id] = "complete"
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        save_results_to_disk(job_id, {"error": str(e)})

def save_results_to_disk(job_id: str, result: dict):
    file_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to disk at {file_path}")
