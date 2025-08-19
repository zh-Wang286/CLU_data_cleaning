# -*- coding: utf-8 -*-
"""Logging configuration for the application."""

from pathlib import Path
import sys
import uuid

from loguru import logger

from src.config import settings


def setup_logging():
    """
    Set up Loguru logger with console and file sinks based on settings.
    Injects a unique run_id into the log context.
    """
    logger.remove()  # Remove default handler

    # Generate a unique run ID for this execution session
    run_id = uuid.uuid4().hex
    settings.run_id = run_id  # Store it in settings for access elsewhere

    # Console Sink
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[run_id]:.8}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,  # Make it process-safe
    )

    # File Sink
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "clu_{time}.log"

    logger.add(
        log_file_path,
        level=settings.log_level.upper(),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[run_id]} | "
            "{process} | {name}:{function}:{line} - {message}"
        ),
        rotation=settings.log_rotate,
        retention=settings.log_retention,
        compression=settings.log_compress if settings.log_compress != "off" else None,
        enqueue=True,  # Make it process-safe
        encoding="utf-8",
    )

    # Add run_id to the global context for all log messages
    logger.configure(extra={"run_id": run_id})

    logger.info(f"Logging initialized with run_id: {run_id}")
    logger.info(f"Log files will be saved in: {log_dir.resolve()}")


# To be called once at the application's entry point.
# setup_logging()
