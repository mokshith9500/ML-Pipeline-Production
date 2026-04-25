"""
utils/logger.py
---------------
Centralized logging configuration for the ML Pipeline.

Sets up both file-based and console logging with consistent formatting.
All pipeline modules import their logger from here to ensure uniform
log structure across every stage.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger instance.

    Sets up dual-output logging:
      - Console (stdout): For real-time visibility during runs
      - File (logs/pipeline_YYYYMMDD.log): Persistent record of every run

    Args:
        name (str): Logger name, typically the module name (__name__).
        log_dir (str): Directory where log files are stored.
        level (str): Logging level — DEBUG, INFO, WARNING, ERROR.

    Returns:
        logging.Logger: Fully configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ------------------------------------------------------------------
    # Formatter: timestamp | level | module | message
    # ------------------------------------------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ------------------------------------------------------------------
    # File Handler — one log file per day
    # ------------------------------------------------------------------
    log_filename = os.path.join(
        log_dir,
        f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # Capture everything to file
    file_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Console Handler — INFO and above to terminal
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_section_header(logger: logging.Logger, title: str) -> None:
    """
    Log a visual section header for readability in terminal and log files.

    Args:
        logger (logging.Logger): Logger to write to.
        title (str): Section title to display.
    """
    border = "=" * 60
    logger.info(border)
    logger.info(f"  {title.upper()}")
    logger.info(border)


def log_section_footer(logger: logging.Logger, status: str = "COMPLETE") -> None:
    """
    Log a visual section footer with status.

    Args:
        logger (logging.Logger): Logger to write to.
        status (str): Status string to display (e.g. COMPLETE, FAILED).
    """
    logger.info(f"  STATUS: {status}")
    logger.info("-" * 60)
