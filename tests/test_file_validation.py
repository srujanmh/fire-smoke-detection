"""
Unit tests for file validation utilities.
Tests magic byte validation, file size limits, and stream URL validation.
"""

import pytest
from utils.file_validation import (
    validate_image_file,
    validate_video_file,
    validate_stream_url,
    sanitize_filename
)
from utils.logging_config import sanitize_stream_url


def test_sanitize_filename():
    assert sanitize_filename("safe_image.jpg") == "safe_image.jpg"
    assert sanitize_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert sanitize_filename("my image (1) #2.png") == "my_image__1___2.png"


def test_validate_image_file_valid_jpeg():
    # Valid JPEG header: \xff\xd8\xff...
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100
    valid, msg = validate_image_file(jpeg_bytes, "sample.jpg", max_size_mb=10)
    assert valid is True
    assert "valid" in msg.lower()


def test_validate_image_file_valid_png():
    # Valid PNG header: \x89PNG\r\n\x1a\n...
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
    valid, msg = validate_image_file(png_bytes, "sample.png", max_size_mb=10)
    assert valid is True


def test_validate_image_file_corrupted_header():
    fake_bytes = b"NOT_AN_IMAGE_HEADER_AT_ALL" + b"\x00" * 50
    valid, msg = validate_image_file(fake_bytes, "sample.jpg", max_size_mb=10)
    assert valid is False
    assert "header does not match" in msg.lower()


def test_validate_image_file_oversized():
    # 5MB test file with 2MB limit
    big_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (3 * 1024 * 1024)
    valid, msg = validate_image_file(big_bytes, "large.jpg", max_size_mb=2)
    assert valid is False
    assert "exceeds limit" in msg.lower()


def test_validate_video_file():
    valid, _ = validate_video_file(10 * 1024 * 1024, "clip.mp4", max_size_mb=50)
    assert valid is True

    invalid_ext, msg = validate_video_file(10 * 1024 * 1024, "clip.exe", max_size_mb=50)
    assert invalid_ext is False
    assert "unsupported" in msg.lower()


def test_validate_stream_url():
    valid, _ = validate_stream_url("rtsp://admin:12345@192.168.1.50:554/h264")
    assert valid is True

    valid_http, _ = validate_stream_url("http://example.com/live.m3u8")
    assert valid_http is True

    invalid, msg = validate_stream_url("ftp://server/stream")
    assert invalid is False


def test_sanitize_stream_url():
    raw_url = "rtsp://alice:SecretPass99@192.168.1.100:554/live"
    sanitized = sanitize_stream_url(raw_url)
    assert "SecretPass99" not in sanitized
    assert "alice" not in sanitized
    assert sanitized == "rtsp://***:***@192.168.1.100:554/live"
