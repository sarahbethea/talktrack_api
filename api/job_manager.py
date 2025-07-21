# Background job runner and state manager
from pipeline.pipeline import run_pipeline

JOB_STATUS = {}
JOB_RESULTS = {}

def run_pipeline_and_store(audio_path: str, job_id: str):
    try:
        JOB_STATUS[job_id] = "processing"
        result = run_pipeline(audio_path)
        JOB_STATUS[job_id] = "complete"
        JOB_RESULTS[job_id] = result
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        JOB_RESULTS[job_id] = {"error": str(e)}
