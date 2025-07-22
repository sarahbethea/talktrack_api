from datetime import datetime, timezone
from config import RESULTS_DIR
import os, json

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results_to_disk(job_id: str, results: dict):
    ensure_results_dir()
    results["created_at"] = datetime.now(timezone.utc).isoformat()
    file_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to disk at {file_path}")


