"""
Unit tests for evaluation module and IoU calculations.
"""

import pytest
from modules.evaluation import calculate_iou, parse_yolo_label_file


def test_calculate_iou_exact_overlap():
    boxA = [10.0, 10.0, 50.0, 50.0]
    boxB = [10.0, 10.0, 50.0, 50.0]
    iou = calculate_iou(boxA, boxB)
    assert iou == pytest.approx(1.0)


def test_calculate_iou_no_overlap():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [20.0, 20.0, 30.0, 30.0]
    iou = calculate_iou(boxA, boxB)
    assert iou == pytest.approx(0.0)


def test_calculate_iou_partial_overlap():
    # Box A: [0, 0, 20, 20], area = 400
    # Box B: [10, 0, 30, 20], area = 400
    # Overlap: [10, 0, 20, 20], area = 200
    # Union: 400 + 400 - 200 = 600
    # IoU: 200 / 600 = 0.33333
    boxA = [0.0, 0.0, 20.0, 20.0]
    boxB = [10.0, 0.0, 30.0, 20.0]
    iou = calculate_iou(boxA, boxB)
    assert iou == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_parse_yolo_label_file(tmp_path):
    label_file = tmp_path / "sample.txt"
    # Format: cls x_c y_c w h (normalized)
    label_file.write_text("0 0.5 0.5 0.2 0.4\n1 0.2 0.3 0.1 0.2\n")

    boxes = parse_yolo_label_file(str(label_file), img_width=1000, img_height=1000)
    assert len(boxes) == 2

    # Box 0: x_c=500, y_c=500, w=200, h=400 -> x1=400, y1=300, x2=600, y2=700
    b0 = boxes[0]
    assert b0["class_name"] == "fire"
    assert b0["box"] == pytest.approx([400.0, 300.0, 600.0, 700.0])

    # Box 1: smoke
    b1 = boxes[1]
    assert b1["class_name"] == "smoke"
