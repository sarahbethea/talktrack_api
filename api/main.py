"""
FastAPI entrypoint for the TalkTrack API.

Run locally:
    uvicorn api.main:app --reload
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import router as api_router
from utils.cleanup import schedule_cleanup, shutdown_event
from utils.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan hook:
      - initialize logging,
      - start periodic cleanup,
      - ensure cleanup thread is signaled on shutdown.
    """
    setup_logging()
    logging.info("Talktrack API started")

    # Start cleanup scheduler (every 10 minutes)
    schedule_cleanup(interval_seconds=600)  # every 10 minutes

    # Hand control to FastAPI to run the app
    yield 

    # Shutdown logic
    logging.info("Shutting down Talktrack API...")
    shutdown_event.set()


app = FastAPI(title="Talktrack API", lifespan=lifespan)
app.include_router(api_router)