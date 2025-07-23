# Manages job state and pipeline execution
from pipeline.pipeline import run_pipeline
from utils.storage import save_results_to_disk
from utils.progress import update_progress
import logging

JOB_STATUS = {}

def run_pipeline_and_store(audio_path: str, job_id: str):
    try:
        JOB_STATUS[job_id] = "processing"
        update_progress(job_id, "starting", 0)
        logging.info(f"[{job_id}] 🚀 Job started")

        results = run_pipeline(audio_path, job_id)

        JOB_STATUS[job_id] = "complete"
        update_progress(job_id, "complete", 100)
        logging.info(f"[{job_id}] ✅ Job completed")
        save_results_to_disk(job_id, results)
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        update_progress(job_id, "error", 100)
        logging.exception(f"[{job_id}] ❌ Job failed: {e}")
        save_results_to_disk(job_id, {"error": str(e)})
        
