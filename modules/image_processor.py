"""
Image processing pipeline for uploaded static images.
Handles file validation, YOLO inference, bounding box overlay,
incident logging, and downloadable asset generation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import io

from modules.detector import FireSmokeDetector
from modules.event_manager import EventManager
from utils.file_validation import validate_image_file, sanitize_filename
from utils.image_utils import bytes_to_cv2, bgr_to_rgb, rgb_to_bgr, draw_detections, cv2_to_bytes
from utils.logging_config import logger


class ImageProcessor:
    """Orchestrates image validation, detection, and annotation."""

    def __init__(self, detector: FireSmokeDetector, event_manager: EventManager):
        self.detector = detector
        self.event_manager = event_manager

    def process_image(
        self,
        file_bytes: bytes,
        filename: str,
        confidence_threshold: Optional[float] = None,
        save_incident: bool = True
    ) -> Dict[str, Any]:
        """
        Process uploaded image bytes through verification, inference, and rendering.
        """
        clean_filename = sanitize_filename(filename)

        # 1. Validation
        is_valid, val_msg = validate_image_file(file_bytes, clean_filename)
        if not is_valid:
            logger.warning(f"Image validation rejected '{clean_filename}': {val_msg}")
            return {
                "success": False,
                "error": val_msg,
                "status": "VALIDATION_ERROR",
                "summary": {}
            }

        # 2. Decode bytes to OpenCV BGR
        cv2_bgr = bytes_to_cv2(file_bytes)
        if cv2_bgr is None:
            return {
                "success": False,
                "error": "Failed to decode image data. Corrupted or invalid image.",
                "status": "DECODE_ERROR",
                "summary": {}
            }

        original_rgb = bgr_to_rgb(cv2_bgr)

        # 3. Inference
        det_result = self.detector.detect(original_rgb, confidence_threshold=confidence_threshold)

        # 4. Render Annotations
        annotated_rgb = draw_detections(original_rgb, det_result.get("detections", []), is_rgb=True)

        # 5. Incident persistence
        incident_info = None
        if save_incident and det_result.get("status") in ["FIRE DETECTED", "SMOKE DETECTED", "FIRE AND SMOKE DETECTED"]:
            incident_info = self.event_manager.process_incident(
                frame=annotated_rgb,
                detection_result=det_result,
                source="image_upload",
                filename=clean_filename,
                force_log=True  # Always log explicit single image uploads
            )

        # 6. Generate downloadable bytes
        annotated_bgr = rgb_to_bgr(annotated_rgb)
        download_bytes = cv2_to_bytes(annotated_bgr, ext=".jpg")

        # 7. Summary
        summary = {
            "total_objects": det_result.get("total_count", 0),
            "fire_count": det_result.get("fire_count", 0),
            "smoke_count": det_result.get("smoke_count", 0),
            "highest_confidence": det_result.get("highest_confidence", 0.0),
            "processing_time_ms": det_result.get("inference_time_ms", 0.0),
            "status": det_result.get("status", "NORMAL"),
            "is_simulated": det_result.get("is_simulated", False)
        }

        # Human-readable message
        if summary["total_objects"] == 0:
            status_message = "No fire or smoke detected above the selected confidence threshold."
        else:
            status_message = f"Detected {summary['fire_count']} fire and {summary['smoke_count']} smoke object(s)."

        return {
            "success": True,
            "error": None,
            "status": det_result.get("status", "NORMAL"),
            "status_message": status_message,
            "summary": summary,
            "original_image": original_rgb,
            "annotated_image": annotated_rgb,
            "download_bytes": download_bytes,
            "detections": det_result.get("detections", []),
            "incident_info": incident_info
        }
