from datetime import datetime, timezone
from config import EXPIRATION_SECONDS, RESULTS_DIR
from storage import ensure_results_dir
import os, threading, time, json

# Might not need this
ensure_results_dir()

def cleanup_old_results():
    now = datetime.now(timezone.utc)
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
        
        file_path = os.path.join(RESULTS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            created_at = datetime.fromisoformat(data.get("created_at", ""))
            age = (now - created_at).total_seconds()

            if age > EXPIRATION_SECONDS:
                os.remove(file_path)
                print(f"[CLEANUP]Removed old results file: {file_path}")

        except Exception as e:
            print(f"[CLEANUP] Error reading {file_path}: {e}")


def schedule_cleanup(interval_seconds: int = 3600):
    def _loop():
        while True:
            print("[CLEANUP] Running scheduled cleanup...")
            cleanup_old_results()
            time.sleep(interval_seconds)
    
    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()