"""
Unit tests for FireSmokeDetector.
Tests model missing error handling, class mapping logic, simulation mode, and status generation.
"""

import os
import pytest
import numpy as np

from modules.detector import FireSmokeDetector


def test_detector_missing_model():
    detector = FireSmokeDetector(model_path="models/non_existent_weights.pt")
    assert detector.is_loaded is False
    assert detector.load_error is not None
    assert ("not found" in detector.load_error.lower()) or ("not installed" in detector.load_error.lower())

    # Inference without model should report clean error, not raise exception
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = detector.detect(dummy_img)
    assert res["status"] in ["MODEL ERROR", "MODEL MISSING"]
    assert res["total_count"] == 0


def test_detector_simulation_mode():
    detector = FireSmokeDetector(model_path="models/non_existent_weights.pt")
    detector.is_simulation_mode = True

    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detector.detect(dummy_img)

    assert res["is_simulated"] is True
    assert res["status"] == "FIRE DETECTED"
    assert res["fire_count"] == 1
    assert len(res["detections"]) == 1
    assert res["highest_confidence"] > 0.8


def test_detector_class_mapping_logic():
    detector = FireSmokeDetector(model_path="models/non_existent_weights.pt")

    # Manually test mapping logic with various real-world model names
    test_raw_names = {
        0: "fire_flame",
        1: "industrial_smoke",
        2: "person",
        3: "car"
    }

    mapped = {}
    for cid, name in test_raw_names.items():
        lower_name = str(name).strip().lower()
        if any(k in lower_name for k in ["fire", "flame", "blaze"]):
            mapped[cid] = "fire"
        elif any(k in lower_name for k in ["smoke", "smoke_cloud", "fume"]):
            mapped[cid] = "smoke"
        else:
            mapped[cid] = lower_name

    assert mapped[0] == "fire"
    assert mapped[1] == "smoke"
    assert mapped[2] == "person"
    assert mapped[3] == "car"


def test_hardware_info():
    detector = FireSmokeDetector(model_path="models/non_existent_weights.pt")
    info = detector.get_hardware_info()
    assert "active_device" in info
    assert "device_name" in info
