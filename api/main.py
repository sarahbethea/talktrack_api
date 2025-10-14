"""
FastAPI entrypoint for the TalkTrack API.

Run locally:
    uvicorn api.main:app --reload
"""
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from api.routes import router as api_router
from utils.cleanup import schedule_cleanup, shutdown_event
from utils.logging_config import configure_logging
from dotenv import load_dotenv
from config import LOGS_DIR
import logging
import os

# Get environment variables from .env
load_dotenv()  

configure_logging(LOGS_DIR)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan hook:
      - initialize logging,
      - start periodic cleanup,
      - ensure cleanup thread is signaled on shutdown.
    """
    logger.info("Talktrack API started")

    # Start cleanup scheduler (every 10 minutes)
    schedule_cleanup(interval_seconds=600)  # every 10 minutes

    # Hand control to FastAPI to run the app
    yield 

    # Shutdown logic
    logger.info("Shutting down Talktrack API...")
    shutdown_event.set()


app = FastAPI(title="Talktrack API", lifespan=lifespan)
app.include_router(api_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)