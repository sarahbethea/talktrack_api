"""
In-memory job progress tracker.

Responsibilities:
- Track per-job progress (stage + percent) in a process-local dict.
- Provide read access for routes/polling endpoints.
"""
import logging

logger = logging.getLogger(__name__)

JOB_PROGRESS = {}

def update_progress(job_id: str, stage: str, percent: int):
    """
    Update progress for a job.

    Args:
        job_id: Unique job identifier.
        stage:  Human-readable step name (e.g., "transcribing", "diarizing").
        percent: Integer 0–100 indicating completion percentage.
    """
    JOB_PROGRESS[job_id] = {"stage": stage, "percent": percent}
    logger.info(f"Updated progress for {job_id}: {stage} at {percent}%")

def get_progress(job_id: str) -> dict:
    """
    Get the latest recorded progress for a job.

    Args:
        job_id: Unique job identifier.

    Returns:
        dict with keys {"stage", "percent"} if present, otherwise {}.
    """
    return JOB_PROGRESS.get(job_id, {})

