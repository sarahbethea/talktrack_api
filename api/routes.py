# api/routes.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from .job_manager import run_pipeline_and_store, JOB_STATUS
from utils.cleanup import cleanup_results_files, cleanup_temp_files
from utils.progress import JOB_PROGRESS
from utils.auth import verify_api_key
from config import RESULTS_DIR
import uuid, os, json

router = APIRouter()

@router.post("/upload-audio") 
# BackgroundTasks is a fastapi feature for running a function after the response is sent
async def upload_audio(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None, 
    user: dict = Depends(verify_api_key)
): 
    print(f"User authenticated: {user['user_id']} with plan {user['plan']}")

    job_id = str(uuid.uuid4())
    file_path = f"temp/{job_id}_{file.filename}.wav"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f: 
        f.write(await file.read()) # file read reads entire uploaded file, wait is used because read() is asynchronous (non-blocking)

    background_tasks.add_task(run_pipeline_and_store, file_path, job_id) # this says to fastapi, "run run_pipeline_and_store in the background once I send the response"

    return {"job_id": job_id} # send job id back to client immediately 


@router.get("/status/{job_id}")
def get_status(job_id: str):
    print(f"[STATUS] Lookup for job_id: {job_id}")
    print("Current progress dict:", JOB_PROGRESS.get(job_id))

    status = JOB_STATUS.get(job_id, "not_found")
    progress = JOB_PROGRESS.get(job_id, {})

    return {
        "status": status, 
        "stage": progress.get("stage", None),
        "progress": progress.get("percent", None)
    }


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
    cleanup_results_files()
    cleanup_temp_files()
    return {"status": "cleanup complete"}



