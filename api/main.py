from fastapi import FastAPI
from api.routes import router as api_router
from utils.cleanup import schedule_cleanup

app = FastAPI(title="PP Extension API")
app.include_router(api_router)

# Start cleanup scheduler (every 10 minutes = 600 seconds)
schedule_cleanup(interval_seconds=600)
