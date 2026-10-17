"""
IP Camera / RTSP Network Stream Module.
Connects to network streams with timeout management,
credential masking, and clean disconnects.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from utils.file_validation import validate_stream_url
from utils.logging_config import logger, sanitize_stream_url
from utils.image_utils import bgr_to_rgb


class IPCameraManager:
    """Manages RTSP/HTTP network video stream captures safely."""

    def __init__(self, stream_url: str = ""):
        self.stream_url = stream_url.strip()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.masked_url = sanitize_stream_url(self.stream_url)
        self.last_frame_time = 0.0
        self.fps_estimate = 0.0
        self.frame_count = 0

    def connect(self, stream_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Connect to the network stream URL.
        Credentials in the URL are never written to plain-text logs.
        """
        if stream_url:
            self.stream_url = stream_url.strip()
            self.masked_url = sanitize_stream_url(self.stream_url)

        # 1. Validation
        is_valid, err_msg = validate_stream_url(self.stream_url)
        if not is_valid:
            logger.warning(f"Invalid stream URL format: {err_msg}")
            return False, err_msg

        self.disconnect()

        logger.info(f"Connecting to network stream: {self.masked_url}...")
        try:
            # Open stream
            self.cap = cv2.VideoCapture(self.stream_url)

            # Check if capture opened
            if not self.cap.isOpened():
                self.is_connected = False
                error_detail = (
                    f"Unable to connect to stream at {self.masked_url}. "
                    "Verify the camera IP, port, stream path, and network connectivity."
                )
                logger.error(error_detail)
                return False, error_detail

            # Test frame read to confirm active data
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.disconnect()
                error_detail = f"Connected to {self.masked_url}, but no video frames were received."
                logger.warning(error_detail)
                return False, error_detail

            self.is_connected = True
            self.frame_count = 1
            self.last_frame_time = time.time()
            logger.info(f"Connected successfully to stream: {self.masked_url}")
            return True, f"Connected successfully to {self.masked_url}"

        except Exception as e:
            self.disconnect()
            error_detail = f"Exception connecting to network stream: {str(e)}"
            logger.error(error_detail, exc_info=True)
            return False, error_detail

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame from the network stream and convert BGR -> RGB.
        """
        if not self.is_connected or self.cap is None or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Stream frame drop on {self.masked_url}")
                return False, None

            now = time.time()
            dt = now - self.last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * instant_fps if self.fps_estimate > 0 else instant_fps
            self.last_frame_time = now
            self.frame_count += 1

            rgb_frame = bgr_to_rgb(frame)
            return True, rgb_frame

        except Exception as e:
            logger.error(f"Error reading frame from stream {self.masked_url}: {e}")
            return False, None

    def disconnect(self) -> None:
        """Release network stream capture resources."""
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info(f"Stream disconnected: {self.masked_url}")
            except Exception as e:
                logger.warning(f"Error disconnecting stream: {e}")
            finally:
                self.cap = None

        self.is_connected = False
        self.fps_estimate = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return stream diagnostics."""
        return {
            "is_connected": self.is_connected,
            "masked_url": self.masked_url,
            "fps": round(self.fps_estimate, 1),
            "frame_count": self.frame_count
        }

    def __del__(self):
        self.disconnect()
