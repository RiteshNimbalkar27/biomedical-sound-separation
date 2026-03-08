"""
logger.py
─────────────────────────────────────────────
Simple logging utility for the project.
Writes to both console and a log file.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "experiments/logs") -> logging.Logger:
    """
    Create and return a logger that writes to both
    the console and a timestamped log file.

    Args:
        name:    Logger name (usually the calling module's __name__).
        log_dir: Directory to store log files.

    Returns:
        Configured Python Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Console handler — force UTF-8 so special characters work on Windows
    console_stream = open(sys.stdout.fileno(), 
                          mode='w', 
                          encoding='utf-8', 
                          buffering=1, 
                          closefd=False)
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # File handler — UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d: %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger