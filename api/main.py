from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import router as api_router
from utils.cleanup import schedule_cleanup, shutdown_event
from utils.logging_config import setup_logging
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting PP Extension API...")
    setup_logging()
    logging.info("PP Extension API started")

    # Start cleanup scheduler
    schedule_cleanup(interval_seconds=600)  # every 10 minutes

    yield # Run the app

    # Shutdown logic
    print("Shutting down PP Extension API...")
    shutdown_event.set()


app = FastAPI(title="PP Extension API", lifespan=lifespan)
app.include_router(api_router)