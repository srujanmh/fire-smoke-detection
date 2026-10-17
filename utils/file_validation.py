"""
File validation utilities for images, videos, and datasets.
Includes MIME/magic-byte checks, filename sanitization, and size limits.
"""

import os
import re
from typing import Tuple, Optional
from pathlib import Path
from utils.logging_config import logger

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Known magic bytes for image formats
IMAGE_MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # RIFF....WEBP
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and special character exploits.
    """
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^\w\.\-_]", "_", clean_name)
    return clean_name or "uploaded_file"


def validate_image_file(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 25
) -> Tuple[bool, str]:
    """
    Validate uploaded image bytes against size limits, file extension, and magic bytes.
    """
    if not file_bytes:
        return False, "Uploaded image is empty (0 bytes)."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Image size ({size_mb:.2f} MB) exceeds limit of {max_size_mb} MB."

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"

    # Magic byte inspection
    header = file_bytes[:16]
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = header.startswith(b"RIFF") and b"WEBP" in header

    if not (is_jpeg or is_png or is_webp):
        return False, "File header does not match a valid image format (corrupted or disguised file)."

    return True, "Image is valid."


def validate_video_file(
    file_path_or_size: int,
    filename: str,
    max_size_mb: int = 150
) -> Tuple[bool, str]:
    """
    Validate uploaded video against file size limits and file extensions.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"Unsupported video extension '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"

    size_mb = file_path_or_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Video size ({size_mb:.2f} MB) exceeds maximum allowed size of {max_size_mb} MB."

    return True, "Video is valid."


def validate_stream_url(url: str) -> Tuple[bool, str]:
    """
    Validate an RTSP/HTTP/HTTPS stream URL.
    """
    if not url or not url.strip():
        return False, "Stream URL cannot be empty."

    clean_url = url.strip()
    valid_schemes = ("rtsp://", "http://", "https://", "rtmp://")
    if not any(clean_url.startswith(scheme) for scheme in valid_schemes):
        return False, f"Invalid stream protocol. Must start with: {', '.join(valid_schemes)}"

    return True, "Stream URL is valid."
