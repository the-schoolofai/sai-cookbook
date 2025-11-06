"""
Centralized Logging Configuration

Provides consistent logging across the application.
"""

import logging
import sys
from pathlib import Path
from app.config.settings import get_settings


def setup_logging() -> logging.Logger:
    """
    Setup application logging configuration.

    Returns:
        Logger instance for the application
    """
    settings = get_settings()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler(
                log_dir / "knowledge_assistant.log",
                encoding="utf-8",
            ),
        ],
    )

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger("knowledge_assistant")
    logger.info("Logging configured successfully")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"knowledge_assistant.{name}")
