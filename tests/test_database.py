"""
Unit tests for DatabaseManager.
Tests SQLite initialization, parameterized insertions, queries, and audits.
"""

import os
import tempfile
import pytest
from modules.database import DatabaseManager


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_database_initialization(temp_db):
    assert os.path.exists(temp_db.db_path)
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "detection_events" in tables
        assert "incident_reviews" in tables


def test_insert_and_retrieve_detection_event(temp_db):
    event_id = temp_db.insert_detection_event(
        source="image_upload",
        detected_class="fire",
        confidence=0.92,
        alert_status="FIRE DETECTED",
        filename="test_fire.jpg",
        bbox_data=[{"box": [10, 10, 50, 50], "class_name": "fire"}],
        processing_duration=45.2
    )
    assert event_id == 1

    events = temp_db.get_events(limit=10)
    assert len(events) == 1
    ev = events[0]
    assert ev["detected_class"] == "fire"
    assert ev["confidence"] == pytest.approx(0.92)
    assert ev["filename"] == "test_fire.jpg"


def test_event_filtering(temp_db):
    temp_db.insert_detection_event(source="webcam", detected_class="fire", confidence=0.85, alert_status="FIRE DETECTED")
    temp_db.insert_detection_event(source="video_upload", detected_class="smoke", confidence=0.75, alert_status="SMOKE DETECTED")

    fire_events = temp_db.get_events(class_filter="fire")
    assert len(fire_events) == 1
    assert fire_events[0]["detected_class"] == "fire"

    video_events = temp_db.get_events(source_filter="video_upload")
    assert len(video_events) == 1
    assert video_events[0]["source"] == "video_upload"


def test_summary_metrics(temp_db):
    temp_db.insert_detection_event(source="image_upload", detected_class="fire", confidence=0.9, alert_status="FIRE DETECTED", filename="img1.jpg")
    temp_db.insert_detection_event(source="image_upload", detected_class="smoke", confidence=0.8, alert_status="SMOKE DETECTED", filename="img2.jpg")

    metrics = temp_db.get_summary_metrics()
    assert metrics["total_events"] == 2
    assert metrics["fire_events"] == 1
    assert metrics["smoke_events"] == 1
    assert metrics["images_analyzed"] == 2


def test_incident_reviews(temp_db):
    event_id = temp_db.insert_detection_event(source="webcam", detected_class="smoke", confidence=0.6, alert_status="SMOKE DETECTED")
    review_id = temp_db.insert_review(event_id=event_id, reviewer_status="false_positive", notes="Steam from coffee mug")
    assert review_id == 1

    summary = temp_db.get_reviews_summary()
    assert summary.get("false_positive") == 1


def test_export_to_csv(temp_db):
    temp_db.insert_detection_event(source="webcam", detected_class="fire", confidence=0.95, alert_status="FIRE DETECTED")
    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        temp_db.export_to_csv(csv_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
