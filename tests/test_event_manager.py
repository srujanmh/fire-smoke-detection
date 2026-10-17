"""
Unit tests for AlertSystem and EventManager.
Tests temporal window confirmation, rate limiting, and alert states.
"""

import os
import time
import tempfile
import numpy as np
import pytest

from modules.alerts import AlertSystem
from modules.database import DatabaseManager
from modules.event_manager import EventManager


@pytest.fixture
def test_setup():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    temp_dir = tempfile.mkdtemp()

    db = DatabaseManager(db_path=db_path)
    alerts = AlertSystem(cooldown_seconds=1.0, temporal_window_size=3, temporal_min_detections=2, audio_enabled=False)
    events = EventManager(db_manager=db, alert_system=alerts, snapshots_dir=temp_dir, log_cooldown_seconds=1.0)

    yield {"db": db, "alerts": alerts, "events": events, "dir": temp_dir}

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except (PermissionError, OSError):
            pass


def test_temporal_confirmation_window(test_setup):
    alerts = test_setup["alerts"]

    # Frame 1: Fire detected once -> Should be UNCERTAIN (window has 1/3, min required is 2)
    res1 = alerts.process_frame_detections({"fire_count": 1, "smoke_count": 0})
    assert res1["state"] == "UNCERTAIN"
    assert res1["confirmed_fire"] is False

    # Frame 2: Fire detected again -> Should become FIRE DETECTED (window has 2/3, min is 2)
    res2 = alerts.process_frame_detections({"fire_count": 1, "smoke_count": 0})
    assert res2["state"] == "FIRE DETECTED"
    assert res2["confirmed_fire"] is True

    # Frame 3: Normal frame (no fire) -> Still 2/3 in window, state remains confirmed
    res3 = alerts.process_frame_detections({"fire_count": 0, "smoke_count": 0})
    assert res3["state"] == "FIRE DETECTED"

    # Frame 4: Another normal frame -> Now 1/3 in window, drops back
    res4 = alerts.process_frame_detections({"fire_count": 0, "smoke_count": 0})
    assert res4["state"] == "NORMAL"


def test_alert_reset(test_setup):
    alerts = test_setup["alerts"]
    alerts.process_frame_detections({"fire_count": 1, "smoke_count": 0})
    alerts.process_frame_detections({"fire_count": 1, "smoke_count": 0})
    assert alerts.current_alert_state == "FIRE DETECTED"

    alerts.reset_state()
    assert alerts.current_alert_state == "NORMAL"
    assert len(alerts.fire_window) == 0


def test_event_manager_deduplication(test_setup):
    events = test_setup["events"]
    db = test_setup["db"]

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det_mock = {
        "status": "FIRE DETECTED",
        "fire_count": 1,
        "smoke_count": 0,
        "highest_confidence": 0.89,
        "detections": [{"box": [10, 10, 50, 50], "class_name": "fire"}]
    }

    # 1st call: window 1/3 -> UNCERTAIN
    events.process_incident(dummy_frame, det_mock, source="webcam")
    # 2nd call: window 2/3 -> Confirmed FIRE DETECTED -> Persists to DB
    out2 = events.process_incident(dummy_frame, det_mock, source="webcam")
    assert out2["event_logged"] is True

    # 3rd call immediately: Still FIRE DETECTED, but cooldown (1.0s) has NOT elapsed -> Deduplicated!
    out3 = events.process_incident(dummy_frame, det_mock, source="webcam")
    assert out3["event_logged"] is False

    # Check only 1 record in database
    records = db.get_events()
    assert len(records) == 1
