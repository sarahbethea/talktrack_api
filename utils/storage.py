"""
Disk storage utilities.

Responsibilities:   
- Ensure RESULTS_DIR exists.
- Save results JSON to RESULTS_DIR/{job_id}.json with a "created_at" timestamp
"""
from datetime import datetime, timezone
from config import RESULTS_DIR
import json
import os
import logging

logger = logging.getLogger(__name__)

def ensure_results_dir() -> None:
    """Create RESULTS_DIR if missing (idempotent)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results_to_disk(job_id: str, results: dict) -> None:
    """
    Persist results for a job as JSON (adds UTC created_at).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results["created_at"] = datetime.now(timezone.utc).isoformat()
    file_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to disk at %s", file_path)

