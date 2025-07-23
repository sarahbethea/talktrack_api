# Manages job state and pipeline execution
from pipeline.pipeline import run_pipeline
from utils.storage import save_results_to_disk
import logging

JOB_STATUS = {}

def run_pipeline_and_store(audio_path: str, job_id: str):
    try:
        JOB_STATUS[job_id] = "processing"
        logging.info(f"[{job_id}] 🚀 Job started")

        results = run_pipeline(audio_path)
        save_results_to_disk(job_id, results)

        JOB_STATUS[job_id] = "complete"
        logging.info(f"[{job_id}] ✅ Job completed")
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        save_results_to_disk(job_id, {"error": str(e)})
        logging.exception(f"[{job_id}] ❌ Job failed: {e}")

