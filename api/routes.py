# api/routes.py
"""
FastAPI routes for TalkTrack API.

Endpoints:
- POST /upload-audio: Accepts an audio file, enqueues background processing, returns job_id immediately.
- GET  /status/{job_id}: Returns coarse job status + progress (if available).
- GET  /results/{job_id}: Returns structured results JSON once ready.
- POST /cleanup: Deletes old temp/result artifacts (consider protecting).
- GET  /ping: Health check.

Notes:
- Minimal validation is applied to the uploaded file (MIME type).
- Uses BackgroundTasks to run the pipeline after responding (non-blocking).
"""
import uuid
import os
import json
import logging

from fastapi import (
    APIRouter, 
    UploadFile, 
    File, 
    BackgroundTasks, 
    Depends
)

from .job_manager import run_pipeline_and_store, JOB_STATUS
from utils.cleanup import cleanup_results_files, cleanup_temp_files
from utils.progress import JOB_PROGRESS
from utils.auth import verify_api_key
from config import RESULTS_DIR

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload-audio") 
async def upload_audio(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None, 
    user: dict = Depends(verify_api_key),
) -> dict[str, str]: 
    """
    Accept an audio file, enqueue processing, and return a job_id immediately.

    Processing runs in the background via BackgroundTasks so the client can
    begin polling /status/{job_id} and later /results/{job_id}.
    """
    logger.info("User authenticated: %s (plan=%s)", user.get("user_id"), user.get("plan"))

    job_id = str(uuid.uuid4())
    file_path = f"temp/{job_id}_{file.filename}.wav"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f: 
        f.write(await file.read()) 

    background_tasks.add_task(run_pipeline_and_store, file_path, job_id) 

    return {"job_id": job_id} # send job id back to client immediately 


@router.get("/status/{job_id}")
def get_status(job_id: str) -> dict:
    """Return current job status and progress (if available)."""
    status = JOB_STATUS.get(job_id, "not_found")
    progress = JOB_PROGRESS.get(job_id, {})
    logger.debug("Status lookup job_id=%s -> %s %s", job_id, status, progress)
    return {
        "status": status, 
        "stage": progress.get("stage", None),
        "progress": progress.get("percent", None)
    }


@router.get("/results/{job_id}")
def get_results(job_id: str) -> dict:
    """Return results JSON if ready; else sentinel error used by client."""
    results_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if not os.path.exists(results_path):
        logger.debug("Results not ready for job_id=%s", job_id)
        return {"error": "not ready"}
    
    with open(results_path, "r", encoding="utf-8") as f:
        # return json read from results file
        data = json.load(f)
        logger.debug("Loaded results for job_id=%s (%d bytes)", job_id, len(json.dumps(data)))
        return data


@router.post("/cleanup")
def trigger_cleanup() -> dict[str, str]:
    """Delete temp/result artifacts (dev utility)."""
    cleanup_results_files()
    cleanup_temp_files()
    logger.info("Cleanup complete")
    return {"status": "cleanup complete"}


@router.get("/ping")
def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "pong"}

