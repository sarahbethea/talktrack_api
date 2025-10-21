"""
Job orchestration for the TalkTrack pipeline.

Responsibilities:
- Track JOB_STATUS for each job_id.
- Run the end-to-end pipeline (transcribe → diarize → summarize/classify).
- Persist results JSON to disk for retrieval by /results/{job_id}.
"""
from pipeline.pipeline import run_pipeline
from utils.storage import save_results_to_disk
from utils.progress import update_progress
import logging

logger = logging.getLogger(__name__)

# Maps job_id -> "queued" | "processing" | "complete" | "failed"
JOB_STATUS = {}

def run_pipeline_and_store(audio_path: str, job_id: str, transcriber, diarizer, analyzer) -> None:
    """
    Execute the pipeline for a single job and store outputs to disk.

    Args:
        audio_path: Filesystem path to the uploaded audio file.
        job_id: Unique identifier for the job (UUID string).

    Side effects:
        - Updates JOB_STATUS[job_id] across stages.
        - Calls update_progress(job_id, stage, percent).
        - Writes results JSON via save_results_to_disk(job_id, results).
    """
    try:
        JOB_STATUS[job_id] = "processing"
        update_progress(job_id, "starting", 0)
        logging.info(f"[{job_id}] 🚀 Job started")

        results = run_pipeline(audio_path, job_id, transcriber, diarizer, analyzer)

        JOB_STATUS[job_id] = "complete"
        update_progress(job_id, "complete", 100)
        logging.info(f"[{job_id}] ✅ Job completed")
        save_results_to_disk(job_id, results)
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        update_progress(job_id, "error", 100)
        logging.exception(f"[{job_id}] ❌ Job failed: {e}")
        save_results_to_disk(job_id, {"error": str(e)})
        
