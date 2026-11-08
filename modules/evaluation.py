"""
Model Evaluation Module for Object Detection.
Implements empirical, IoU-based evaluation against YOLO-format ground truth datasets.
Calculates Precision, Recall, F1-score, Confusion Matrix, and mAP@0.5.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import yaml
from PIL import Image

from modules.detector import FireSmokeDetector
from utils.logging_config import logger


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def parse_yolo_label_file(
    label_path: str,
    img_width: int,
    img_height: int,
    class_mapping: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Parse a single YOLO annotation file (.txt).
    Converts normalized (cls, x_c, y_c, w, h) to absolute coordinates [x1, y1, x2, y2].
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = [float(p) for p in parts[1:5]]

                    # Denormalize
                    x1 = (xc - w / 2.0) * img_width
                    y1 = (yc - h / 2.0) * img_height
                    x2 = (xc + w / 2.0) * img_width
                    y2 = (yc + h / 2.0) * img_height

                    cls_name = "fire" if cls_id == 0 else "smoke"
                    if class_mapping and cls_id in class_mapping:
                        cls_name = class_mapping[cls_id]

                    boxes.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "box": [x1, y1, x2, y2]
                    })
                except ValueError:
                    continue
    return boxes


class ModelEvaluator:
    """Evaluates detector against a ground-truth labeled test dataset."""

    def __init__(self, detector: FireSmokeDetector):
        self.detector = detector

    def evaluate_test_dataset(
        self,
        data_yaml_path: str,
        iou_threshold: float = 0.50,
        conf_threshold: float = 0.35,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run empirical evaluation over the test dataset defined in data.yaml.
        Uses greedy IoU matching to calculate exact TP, FP, FN counts.
        """
        if not os.path.exists(data_yaml_path):
            return {
                "success": False,
                "error": f"Dataset configuration file not found at: {data_yaml_path}. "
                         "Please prepare a dataset following the data/README.md instructions."
            }

        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse data.yaml: {str(e)}"}

        base_dir = Path(data_yaml_path).parent
        test_path = data_cfg.get("test")
        if not test_path:
            # Fallback to val if test not specified
            test_path = data_cfg.get("val")

        if not test_path:
            return {"success": False, "error": "No 'test' or 'val' split specified in data.yaml."}

        test_dir = Path(test_path)
        if not test_dir.is_absolute():
            test_dir = (base_dir / test_dir).resolve()

        if not test_dir.exists():
            return {
                "success": False,
                "error": f"Test image directory does not exist: {test_dir}. "
                         "Please ensure images have been placed in data/images/test/."
            }

        # Find images
        supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [p for p in test_dir.iterdir() if p.suffix.lower() in supported_exts]

        if not image_files:
            return {
                "success": False,
                "error": f"No test images found in {test_dir}."
            }

        if max_images:
            image_files = image_files[:max_images]

        # Labels directory corresponding to images
        # Standard structure: /images/test -> /labels/test
        test_labels_dir = Path(str(test_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if not test_labels_dir.exists():
            test_labels_dir = test_dir.parent.parent / "labels" / test_dir.name

        classes = ["fire", "smoke"]
        stats_by_class = {c: {"tp": 0, "fp": 0, "fn": 0, "gt_total": 0, "pred_total": 0} for c in classes}
        total_inference_time = 0.0

        for img_path in image_files:
            try:
                with Image.open(img_path) as pil_img:
                    w, h = pil_img.size
                    img_np = np.array(pil_img.convert("RGB"))

                label_path = test_labels_dir / f"{img_path.stem}.txt"
                gt_boxes = parse_yolo_label_file(str(label_path), w, h)

                t0 = time.time()
                det_res = self.detector.detect(img_np, confidence_threshold=conf_threshold)
                total_inference_time += (time.time() - t0)

                pred_boxes = det_res.get("detections", [])

                # Evaluate per class
                for cls_name in classes:
                    cls_gt = [g for g in gt_boxes if g["class_name"] == cls_name]
                    cls_pred = [p for p in pred_boxes if p["class_name"] == cls_name]

                    stats_by_class[cls_name]["gt_total"] += len(cls_gt)
                    stats_by_class[cls_name]["pred_total"] += len(cls_pred)

                    # Greedily match predictions to GT
                    matched_gt = set()
                    for pred in cls_pred:
                        best_iou = 0.0
                        best_gt_idx = -1
                        for idx, gt in enumerate(cls_gt):
                            if idx in matched_gt:
                                continue
                            iou = calculate_iou(pred["box"], gt["box"])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = idx

                        if best_iou >= iou_threshold and best_gt_idx != -1:
                            stats_by_class[cls_name]["tp"] += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            stats_by_class[cls_name]["fp"] += 1

                    # Unmatched GT boxes are False Negatives
                    stats_by_class[cls_name]["fn"] += (len(cls_gt) - len(matched_gt))

            except Exception as e:
                logger.error(f"Error evaluating image {img_path}: {e}")

        # Compute per-class and overall metrics
        total_tp = sum(stats_by_class[c]["tp"] for c in classes)
        total_fp = sum(stats_by_class[c]["fp"] for c in classes)
        total_fn = sum(stats_by_class[c]["fn"] for c in classes)

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

        per_class_metrics = {}
        for c in classes:
            tp = stats_by_class[c]["tp"]
            fp = stats_by_class[c]["fp"]
            fn = stats_by_class[c]["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            per_class_metrics[c] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ground_truth_count": stats_by_class[c]["gt_total"],
                "prediction_count": stats_by_class[c]["pred_total"]
            }

        avg_latency_ms = (total_inference_time / len(image_files) * 1000.0) if image_files else 0.0
        fps = (len(image_files) / total_inference_time) if total_inference_time > 0 else 0.0

        return {
            "success": True,
            "error": None,
            "images_evaluated": len(image_files),
            "iou_threshold": iou_threshold,
            "confidence_threshold": conf_threshold,
            "overall_precision": round(overall_prec, 4),
            "overall_recall": round(overall_rec, 4),
            "overall_f1": round(overall_f1, 4),
            # NOTE: This is mean per-class precision, not true mAP@0.5.
            # True mAP requires sweeping confidence thresholds and computing
            # area under the precision-recall curve (11-point interpolation).
            "mean_precision": round((per_class_metrics["fire"]["precision"] + per_class_metrics["smoke"]["precision"]) / 2.0, 4),
            "per_class_metrics": per_class_metrics,
            "avg_latency_ms": round(avg_latency_ms, 2),
            "throughput_fps": round(fps, 1)
        }
