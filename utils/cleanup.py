"""
Periodic cleanup utilities for results and temp files.

Responsibilities:
- Start a daemon thread that runs cleanup on a fixed interval.
- Remove expired result JSONs based on their internal "created_at" timestamp.
- Remove expired temp files based on filesystem last-modified time.

Notes:
- Expiration window and directories come from config: EXPIRATION_SECONDS, RESULTS_DIR, TEMP_DIR.
- The cleanup loop can be stopped via `shutdown_event.set()`.
- Minimal/MVP: timestamp parsing assumes ISO 8601 compatible strings.
"""
from datetime import datetime, timezone
from config import EXPIRATION_SECONDS, RESULTS_DIR, TEMP_DIR
from utils.storage import ensure_results_dir
import threading
import logging
import os
import time
import json

logger = logging.getLogger(__name__)

# Ensure result directory exists
ensure_results_dir()

# Signal to stop background cleanup thread
shutdown_event = threading.Event()

def schedule_cleanup(interval_seconds: int = 3600):
    """
    Launch a background (daemon) thread that periodically runs cleanup.

    Args:
        interval_seconds: Seconds between cleanup passes (default: 1 hour).
    """
    def _loop():
        while not shutdown_event.is_set():
            logger.info("[CLEANUP] Running scheduled cleanup...")
            cleanup_results_files()
            cleanup_temp_files()
            shutdown_event.wait(timeout=interval_seconds) # Pauses thread for interval_seconds unless shutdown_event is set

        logger.info("[CLEANUP] Shutdown event triggered — exiting cleanup thread.")

    thread = threading.Thread(target=_loop, daemon=True) # Threads must be passed a callable (_loop)
    thread.start()


def cleanup_results_files():
    """
    Delete expired result JSON files.

    Rule:
        - Load each *.json in RESULTS_DIR.
        - Expect a top-level "created_at" ISO-8601 string.
        - If file age (now - created_at) > EXPIRATION_SECONDS, delete it.
    """
    now = datetime.now(timezone.utc)
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
        
        file_path = os.path.join(RESULTS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            created_at_str = data.get("created_at")
            if not created_at_str:
                logger.info(f"[CLEANUP] Skipping file with no timestamp: {file_path}")
                continue  # ✅ skip files with no timestamp

            created_at = datetime.fromisoformat(created_at_str)
            age = (now - created_at).total_seconds()

            if age > EXPIRATION_SECONDS:
                os.remove(file_path)
                logger.info(f"[CLEANUP] Removed old results file: {file_path}")

        except Exception as e:
            logger.exception(f"[CLEANUP] Error reading {file_path}: {e}")


def cleanup_temp_files():
    """
    Delete expired temp files.

    Rule:
        - For each file in TEMP_DIR, compute age via last-modified time (st_mtime).
        - If age > EXPIRATION_SECONDS, delete it.
    """
    now = datetime.now(timezone.utc)
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            status = os.stat(file_path)
            modified = datetime.fromtimestamp(status.st_mtime, timezone.utc) # st_mtime → last modified time (UNIX timestamp)
            age = (now - modified).total_seconds()

            if age > EXPIRATION_SECONDS:
                os.remove(file_path)
                logger.info(f"[CLEANUP] Removed old temp file: {file_path}")

        except Exception as e:
            logger.exception(f"[CLEANUP] Error removing {file_path}: {e}")

