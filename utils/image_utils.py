"""
Image processing utilities for OpenCV and PIL.
Handles drawing bounding boxes, status badges, saving incident snapshots,
and image conversions.
"""

import os
import cv2
import time
import base64
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from utils.logging_config import logger

# BGR Color definitions for OpenCV
COLOR_FIRE_BGR = (68, 68, 239)       # #EF4444 Red in BGR
COLOR_SMOKE_BGR = (11, 158, 245)     # #F59E0B Amber in BGR
COLOR_UNCERTAIN_BGR = (255, 191, 0)  # Cyan/Blue in BGR
COLOR_DEFAULT_BGR = (200, 200, 200)  # Gray in BGR
COLOR_TEXT_BGR = (255, 255, 255)     # White


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image array to RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image array to BGR for OpenCV saving/drawing."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bytes_to_cv2(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes into an OpenCV BGR numpy array."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image bytes: {e}")
        return None


def cv2_to_bytes(image_bgr: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode OpenCV BGR image into byte buffer."""
    success, encoded_img = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError("Failed to encode cv2 image into bytes.")
    return encoded_img.tobytes()


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    is_rgb: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Annotate image with high-contrast bounding boxes, labels, and confidence tags.
    Works seamlessly on both RGB and BGR images (preserves input color space).
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for det in detections:
        box = det.get("box", [])  # [x1, y1, x2, y2]
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        cls_name = det.get("class_name", "").lower()
        conf = det.get("confidence", 0.0)

        # Determine color
        if "fire" in cls_name or "flame" in cls_name:
            color = (239, 68, 68) if is_rgb else COLOR_FIRE_BGR
            label_text = f"FIRE {conf:.1%}"
        elif "smoke" in cls_name:
            color = (245, 158, 11) if is_rgb else COLOR_SMOKE_BGR
            label_text = f"SMOKE {conf:.1%}"
        else:
            color = (59, 130, 246) if is_rgb else COLOR_UNCERTAIN_BGR
            label_text = f"{cls_name.upper()} {conf:.1%}"

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Draw filled badge for text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(width, height) / 1000.0)
        font_thickness = max(1, int(font_scale * 2))

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        badge_y1 = max(0, y1 - text_h - baseline - 4)
        badge_y2 = y1

        cv2.rectangle(
            annotated,
            (x1, badge_y1),
            (x1 + text_w + 6, badge_y2),
            color,
            -1  # Filled
        )

        # Draw text label in white
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated


def save_incident_snapshot(
    image: np.ndarray,
    output_dir: str,
    prefix: str = "incident",
    is_rgb: bool = True
) -> str:
    """
    Save an annotated incident snapshot with timestamp overlay.
    Returns the absolute path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{timestamp_str}.jpg"
    filepath = os.path.join(output_dir, filename)

    img_to_save = rgb_to_bgr(image) if is_rgb else image.copy()

    # Stamp forensic watermark on bottom-left
    watermark_text = f"INCIDENT LOG: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.putText(
        img_to_save,
        watermark_text,
        (15, img_to_save.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        img_to_save,
        watermark_text,
        (15, img_to_save.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imwrite(filepath, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"Saved incident snapshot to: {filepath}")
    return filepath


def get_image_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64 data URI."""
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
