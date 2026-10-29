"""
Core AI Detection Module for Fire & Smoke Detection System.
Wraps Ultralytics YOLO with dynamic class mapping, device selection,
confidence filtering, and robust error handling.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from utils.logging_config import logger


class FireSmokeDetector:
    """
    Dedicated AI detector for Fire and Smoke using Ultralytics YOLO models.
    """

    def __init__(
        self,
        model_path: str = "models/fire_smoke.pt",
        confidence_threshold: float = 0.25,
        fire_threshold: Optional[float] = None,
        smoke_threshold: Optional[float] = None,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fire_threshold = fire_threshold if fire_threshold is not None else 0.25
        self.smoke_threshold = smoke_threshold if smoke_threshold is not None else 0.45
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        self.model: Optional[Any] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.class_mapping: Dict[int, str] = {}  # id -> canonical name ('fire', 'smoke', 'other')
        self.model_raw_names: Dict[int, str] = {}

        # Device determination
        self.device = self._select_device(device)
        self.is_simulation_mode: bool = False

        # Attempt to load model on creation
        self.load_model()

    def _select_device(self, preferred_device: Optional[str] = None) -> str:
        """Determine whether to use CUDA GPU or CPU."""
        if preferred_device:
            return preferred_device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {gpu_name}")
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA check failed, falling back to CPU: {e}")
                return "cpu"
        return "cpu"

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return information about PyTorch and GPU availability."""
        info = {
            "torch_available": TORCH_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "cuda_available": False,
            "device_name": "CPU",
            "active_device": self.device
        }
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["cuda_available"] = True
            try:
                info["device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["device_name"] = "CUDA Device"
        return info

    def load_model(self, force_reload: bool = False) -> Tuple[bool, str]:
        """
        Load YOLO model from self.model_path and inspect class metadata.
        """
        if not ULTRALYTICS_AVAILABLE:
            self.is_loaded = False
            self.load_error = "Ultralytics package is not installed."
            logger.error(self.load_error)
            return False, self.load_error

        if not os.path.exists(self.model_path):
            self.is_loaded = False
            self.load_error = (
                f"Model file not found at: {self.model_path}. "
                "Please place a trained fire & smoke model (.pt) in the models/ directory "
                "or specify the path in Settings."
            )
            logger.warning(self.load_error)
            return False, self.load_error

        try:
            logger.info(f"Loading YOLO model from: {self.model_path} onto {self.device}...")
            self.model = YOLO(self.model_path)
            # Send model to device
            if hasattr(self.model, "to"):
                self.model.to(self.device)

            # Inspect class names
            raw_names = getattr(self.model, "names", {})
            if isinstance(raw_names, list):
                raw_names = {i: name for i, name in enumerate(raw_names)}
            self.model_raw_names = raw_names

            # Map raw class names to canonical categories
            self.class_mapping = {}
            for class_id, name in self.model_raw_names.items():
                lower_name = str(name).strip().lower()
                if any(k in lower_name for k in ["fire", "flame", "blaze"]):
                    self.class_mapping[class_id] = "fire"
                elif any(k in lower_name for k in ["smoke", "smoke_cloud", "fume"]):
                    self.class_mapping[class_id] = "smoke"
                else:
                    self.class_mapping[class_id] = lower_name

            self.is_loaded = True
            self.load_error = None
            logger.info(f"Model loaded successfully. Class mappings: {self.class_mapping}")
            return True, "Model loaded successfully"

        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load model weights: {str(e)}"
            logger.error(self.load_error, exc_info=True)
            return False, self.load_error

    def unload_model(self) -> None:
        """Release model and clear GPU cache if applicable."""
        self.model = None
        self.is_loaded = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Model unloaded.")

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        is_rgb: bool = False
    ) -> Dict[str, Any]:
        """
        Run object detection on a single image.
        Accepts BGR (default for OpenCV) or RGB (set is_rgb=True).
        Ensures Ultralytics YOLO receives proper BGR array (Ultralytics internally flips BGR -> RGB).
        Returns detection list, status, counts, and inference duration.
        """
        start_time = time.time()
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Guarantee real AI inference whenever weights are loaded
        if not self.is_loaded:
            if not self.is_simulation_mode:
                return {
                    "status": "MODEL ERROR" if self.load_error else "MODEL MISSING",
                    "error": self.load_error or "Model is not loaded.",
                    "detections": [],
                    "fire_count": 0,
                    "smoke_count": 0,
                    "total_count": 0,
                    "highest_confidence": 0.0,
                    "inference_time_ms": 0.0,
                    "is_simulated": False
                }
            else:
                # Controlled simulation mode only if weights are missing
                return self._simulate_detection(image, start_time)

        try:
            # Prepare BGR representation for Ultralytics YOLO and color verification
            if is_rgb:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Run YOLO inference with sensitive base cutoff, then apply calibrated adaptive gates
            yolo_conf = min(conf_thresh, 0.12)
            results = self.model.predict(
                source=image_bgr,
                conf=yolo_conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            detections: List[Dict[str, Any]] = []
            fire_count = 0
            smoke_count = 0
            highest_conf = 0.0

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)

                    img_h, img_w = image_bgr.shape[:2]

                    for box, conf, cls_id in zip(xyxy, confs, classes):
                        canonical_cls = self.class_mapping.get(cls_id, "unknown")
                        raw_name = str(self.model_raw_names.get(cls_id, str(cls_id))).strip().lower()

                        # Process Fire candidates (including dataset class 1 'default' when confirmed as flame)
                        if canonical_cls == "fire" or raw_name in ["fire", "flame", "blaze", "default"]:
                            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                            x2, y2 = min(img_w, int(box[2])), min(img_h, int(box[3]))
                            box_crop = image_bgr[y1:y2, x1:x2]
                            if box_crop.size == 0:
                                continue

                            hsv_crop = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
                            # Pure flame hues in OpenCV HSV (H: 0-180): 0-25 (red/orange) or 165-180 (crimson)
                            m1 = (hsv_crop[:, :, 0] <= 25) & (hsv_crop[:, :, 1] >= 40) & (hsv_crop[:, :, 2] >= 70)
                            m2 = (hsv_crop[:, :, 0] >= 165) & (hsv_crop[:, :, 1] >= 40) & (hsv_crop[:, :, 2] >= 70)
                            warm_ratio = (m1 | m2).sum() / (box_crop.shape[0] * box_crop.shape[1])

                            # If candidate is from generic 'default' dataset label, require high flame density
                            if raw_name == "default" and warm_ratio < 0.25:
                                continue

                            # Adaptive threshold: high combustion density relaxes threshold to 0.14 for macro blazes
                            req_thresh = 0.14 if warm_ratio >= 0.20 else self.fire_threshold
                            if conf < req_thresh or warm_ratio < 0.02:
                                continue

                            canonical_cls = "fire"

                        elif canonical_cls == "smoke" or raw_name in ["smoke", "smoke_cloud", "fume"]:
                            box_w = max(0.0, float(box[2] - box[0]))
                            box_h = max(0.0, float(box[3] - box[1]))
                            # Spatial sanity: ignore ambient wall glare covering >85% of frame
                            if (box_w * box_h) > (0.85 * img_w * img_h):
                                continue
                            if conf < self.smoke_threshold:
                                continue

                            canonical_cls = "smoke"
                        else:
                            continue

                        if conf > highest_conf:
                            highest_conf = float(conf)

                        if canonical_cls == "fire":
                            fire_count += 1
                        elif canonical_cls == "smoke":
                            smoke_count += 1

                        detections.append({
                            "box": [float(x) for x in box],  # [x1, y1, x2, y2]
                            "confidence": float(conf),
                            "class_id": int(cls_id),
                            "class_name": canonical_cls,
                            "raw_class_name": raw_name
                        })

            # Determine overall detection status
            if fire_count > 0 and smoke_count > 0:
                status = "FIRE AND SMOKE DETECTED"
            elif fire_count > 0:
                status = "FIRE DETECTED"
            elif smoke_count > 0:
                status = "SMOKE DETECTED"
            elif len(detections) > 0:
                status = "UNCERTAIN"
            else:
                status = "NORMAL"

            duration_ms = (time.time() - start_time) * 1000.0

            return {
                "status": status,
                "error": None,
                "detections": detections,
                "fire_count": fire_count,
                "smoke_count": smoke_count,
                "total_count": len(detections),
                "highest_confidence": highest_conf,
                "inference_time_ms": duration_ms,
                "is_simulated": False
            }

        except Exception as e:
            logger.error(f"Inference error during detect(): {e}", exc_info=True)
            return {
                "status": "MODEL ERROR",
                "error": str(e),
                "detections": [],
                "fire_count": 0,
                "smoke_count": 0,
                "total_count": 0,
                "highest_confidence": 0.0,
                "inference_time_ms": (time.time() - start_time) * 1000.0,
                "is_simulated": False
            }

    def _simulate_detection(self, image: np.ndarray, start_time: float) -> Dict[str, Any]:
        """
        Controlled demonstration mode strictly labeled as simulated non-AI.
        Used to test UI widgets, alert sounds, and database writes without weights.
        """
        duration_ms = (time.time() - start_time) * 1000.0
        h, w = image.shape[:2]
        # Generate a sample simulated fire box in the center for testing
        sim_box = [w * 0.35, h * 0.35, w * 0.65, h * 0.70]
        return {
            "status": "FIRE DETECTED",
            "error": None,
            "detections": [
                {
                    "box": sim_box,
                    "confidence": 0.88,
                    "class_id": 0,
                    "class_name": "fire",
                    "raw_class_name": "fire_simulated"
                }
            ],
            "fire_count": 1,
            "smoke_count": 0,
            "total_count": 1,
            "highest_confidence": 0.88,
            "inference_time_ms": duration_ms,
            "is_simulated": True
        }
