# api/routes.py
# This file defines the API routes for handling audio uploads and job management.
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from .job_manager import run_pipeline_and_store, JOB_STATUS
from utils.cleanup import cleanup_old_results
from config import RESULTS_DIR
import uuid, os, json

router = APIRouter()

@router.post("/upload-audio") # route decorator, will be triggered whenever post request hits this endpoint
#using async here allows this route to wait for file uploads to finish and frees up server to handle other requests during that wait
# UploadFile is a special fastapi class used to represent uploaded files
# File(...) is a fastapi function and the ... means this is required field. All together, 
# this line says "expect a file named file in the form data of this request and treat it as an UploadFile"
# BackgroundTasks is a fastapi feature for running a function after the response is sent
async def upload_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None): 
    job_id = str(uuid.uuid4())
    file_path = f"temp/{job_id}_{file.filename}.wav"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f: #wb means write binary, since audio files are binary not text
        f.write(await file.read()) # file read reads entire uploaded file, wait is used because read() is asynchronous (non-blocking)

    background_tasks.add_task(run_pipeline_and_store, file_path, job_id) # this says to fastapi, "run run_pipeline_and_store in the background once I send the response"

    return {"job_id": job_id} # send job id back to client immediately 


@router.get("/status/{job_id}")
def get_status(job_id: str):
    return {"status": JOB_STATUS.get(job_id, "not_found")}


@router.get("/results/{job_id}")
def get_results(job_id: str):
    results_path = os.path.join(RESULTS_DIR, f"{job_id}.json")

    if not os.path.exists(results_path):
        return {"error": "not ready"}
    
    with open(results_path, "r", encoding="utf-8") as f:
        # return json read from results file 
        return json.load(f)
    

@router.post("/cleanup")
def trigger_cleanup():
    cleanup_old_results()
    return {"status": "cleanup complete"}



