"""
Camera capture and video stream management.
Wraps OpenCV VideoCapture with device index selection,
FPS estimation, latency tracking, and robust resource cleanup.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from utils.logging_config import logger
from utils.image_utils import bgr_to_rgb


class CameraManager:
    """
    Manages local webcam capture sessions with resource safety
    and frame-rate statistics.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.last_frame_time = 0.0
        self.fps_estimate = 0.0
        self.frame_count = 0

    def start(self, camera_index: Optional[int] = None) -> Tuple[bool, str]:
        """
        Open the camera device. Releases any existing capture first.
        """
        if camera_index is not None:
            self.camera_index = camera_index

        self.stop()  # Clean up existing

        logger.info(f"Opening camera index {self.camera_index}...")
        try:
            # On Windows, cv2.CAP_DSHOW provides fast startup without 10-second latency
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                error_msg = f"Unable to open camera at index {self.camera_index}. Check device connection and permissions."
                logger.error(error_msg)
                return False, error_msg

            self.is_active = True
            self.frame_count = 0
            self.last_frame_time = time.time()
            logger.info(f"Camera {self.camera_index} opened successfully.")
            return True, "Camera initialized successfully."

        except Exception as e:
            self.is_active = False
            error_msg = f"Exception opening camera: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera and convert BGR -> RGB.
        Returns (success_flag, rgb_frame_array).
        """
        if not self.is_active or self.cap is None or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Camera frame read failed or end of stream.")
                return False, None

            # Calculate instantaneous FPS
            now = time.time()
            dt = now - self.last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * instant_fps if self.fps_estimate > 0 else instant_fps
            self.last_frame_time = now
            self.frame_count += 1

            # Convert BGR to RGB
            rgb_frame = bgr_to_rgb(frame)
            return True, rgb_frame

        except Exception as e:
            logger.error(f"Error reading frame from camera: {e}")
            return False, None

    def stop(self) -> None:
        """Release camera hardware resources."""
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info(f"Camera index {self.camera_index} released.")
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            finally:
                self.cap = None

        self.is_active = False
        self.fps_estimate = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return camera performance and status metrics."""
        return {
            "is_active": self.is_active,
            "camera_index": self.camera_index,
            "fps": round(self.fps_estimate, 1),
            "frame_count": self.frame_count
        }

    def __del__(self):
        self.stop()
