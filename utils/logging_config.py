"""
Logging configuration utilities.

Responsibilities:
- Ensure the logs directory exists.
- Configure app logging to write to console and a rotating app.log.
- Respect LOG_LEVEL (your app) and LIB_LOG_LEVEL (libraries) from the environment.
"""
from logging.config import dictConfig
import os

def configure_logging(logs_dir: str) -> None:
    """
    Initialize logging for the application.

    Args:
        logs_dir: Directory where app.log will be created.

    Behavior:
        - Uses LOG_LEVEL for your app packages (api.*, utils.*).
        - Uses LIB_LOG_LEVEL for the root (libraries).
        - Console + rotating file handler share the same simple format.
    """
    os.makedirs(logs_dir, exist_ok=True)
    app_log = os.path.abspath(os.path.join(logs_dir, "app.log"))

    # Read levels from env
    app_level = os.getenv("LOG_LEVEL", "INFO").upper()       
    lib_level = os.getenv("LIB_LOG_LEVEL", "INFO").upper()   # controls libs (root)

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "app_fmt": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            }
        },

        "handlers": {
            "file_app": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": app_level,                
                "filename": app_log,
                "maxBytes": 5_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "formatter": "app_fmt",
            },
            "console_app": {
                "class": "logging.StreamHandler",
                "level": app_level,
                "formatter": "app_fmt",
            },
        },

        "root": {
            "level": lib_level,                 
            "handlers": ["console_app", "file_app"],
        },

        "loggers": {
            "api":   {"level": app_level, "handlers": [], "propagate": True},
            "utils": {"level": app_level, "handlers": [], "propagate": True},

        },
    })
