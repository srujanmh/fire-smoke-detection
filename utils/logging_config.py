"""
Logging configuration for Fire & Smoke Detection System.
Provides thread-safe logging with console and file handlers,
and automatic credential masking for IP camera URLs.
"""

import os
import re
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Regex pattern to match usernames/passwords in URLs: scheme://user:pass@host
CREDENTIAL_URL_PATTERN = re.compile(r'(://)([^:/\s]+):([^@/\s]+)@')


class CredentialSanitizingFormatter(logging.Formatter):
    """Custom formatter that masks credentials in stream URLs or messages."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return CREDENTIAL_URL_PATTERN.sub(r'\1***:***@', original)


def sanitize_stream_url(url: str) -> str:
    """Mask credentials in RTSP/HTTP URLs for safe display and logging."""
    if not url:
        return ""
    return CREDENTIAL_URL_PATTERN.sub(r'\1***:***@', url)


def setup_logger(
    name: str = "fire_smoke_detection",
    log_file: Optional[str] = "app.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up and returns a configured logger instance.
    Prevents duplicate handlers when called repeatedly.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    log_format = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    formatter = CredentialSanitizingFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to initialize file log handler: {e}")

    return logger


# Global logger instance
logger = setup_logger()
