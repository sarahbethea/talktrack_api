# Asynchronous job handling

# POST /upload-audio
# → save audio, start background job
# ← return job_id = abc123

# GET /status/abc123
# → {"status": "processing", "progress": 27}

# GET /results/abc123
# → return full results when ready


from fastapi import APIRouter, UploadFile, File, BackgroundTasks
import uuid, os
from .job_manager import run_pipeline_and_store, JOB_STATUS, JOB_RESULTS
