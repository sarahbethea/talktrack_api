from config import LOGS_DIR
import logging 
import os

os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logging():
    print("Setting up logging...")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # Logs to file, mode="a" appends to existing file
            logging.FileHandler(f"{LOGS_DIR}/app.log", mode='a', encoding='utf-8'), 
            # Logs to console
            logging.StreamHandler() 
        ]
    )
