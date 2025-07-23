JOB_PROGRESS = {}

def update_progress(job_id: str, stage: str, percent: int):
    JOB_PROGRESS[job_id] = {"stage": stage, "percent": percent}
    print(f"[DEBUG] Updated progress for {job_id}: {stage} at {percent}%")  # Debug log for progress updates

def get_progress(job_id: str) -> dict:
    return JOB_PROGRESS.get(job_id, {})

