"""
Event Manager Module for Fire & Smoke Detection.
Coordinates detection outcomes, temporal alerts, snapshot persistence,
and database deduplication logic.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from modules.database import DatabaseManager
from modules.alerts import AlertSystem
from utils.image_utils import save_incident_snapshot
from utils.logging_config import logger


class EventManager:
    """
    Manages detection events, intelligent deduplication, and forensic snapshots.
    Prevents flood of duplicate database records during continuous camera feeds.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        alert_system: AlertSystem,
        snapshots_dir: str = "outputs/snapshots",
        log_cooldown_seconds: float = 5.0
    ):
        self.db = db_manager
        self.alert_system = alert_system
        self.snapshots_dir = snapshots_dir
        self.log_cooldown_seconds = log_cooldown_seconds

        # Track last log timestamp per key: (source, class_category)
        self.last_logged_events: Dict[str, float] = {}

    def process_incident(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, Any],
        source: str,
        filename: Optional[str] = None,
        video_timestamp: Optional[float] = None,
        force_log: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate frame against alert rules and persist event to database & snapshot disk
        if cooldown threshold has elapsed.
        """
        # Run alert system's temporal confirmation
        alert_info = self.alert_system.process_frame_detections(detection_result)
        alert_state = alert_info["state"]

        # Only log confirmed detections or explicit triggers
        is_incident = alert_state in ["FIRE DETECTED", "SMOKE DETECTED", "FIRE AND SMOKE DETECTED"]

        event_id = None
        snapshot_path = None
        now = time.time()
        dedup_key = f"{source}_{alert_state}"
        last_logged = self.last_logged_events.get(dedup_key, 0.0)

        should_persist = force_log or (is_incident and (now - last_logged >= self.log_cooldown_seconds))

        if should_persist:
            try:
                # 1. Save forensic snapshot to disk
                cls_prefix = "fire" if "FIRE" in alert_state else "smoke"
                snapshot_path = save_incident_snapshot(
                    image=frame,
                    output_dir=self.snapshots_dir,
                    prefix=f"{source}_{cls_prefix}",
                    is_rgb=True
                )

                # 2. Insert record into SQLite database
                primary_class = "fire_and_smoke" if "AND" in alert_state else ("fire" if "FIRE" in alert_state else "smoke")
                event_id = self.db.insert_detection_event(
                    source=source,
                    detected_class=primary_class,
                    confidence=detection_result.get("highest_confidence", 0.0),
                    alert_status=alert_state,
                    filename=filename,
                    bbox_data=detection_result.get("detections", []),
                    video_timestamp=video_timestamp,
                    snapshot_path=snapshot_path,
                    processing_duration=detection_result.get("inference_time_ms", 0.0)
                )

                self.last_logged_events[dedup_key] = now
                logger.info(
                    f"Logged incident #{event_id} ({alert_state}) from {source} "
                    f"with confidence {detection_result.get('highest_confidence', 0.0):.2f}"
                )

            except Exception as e:
                logger.error(f"Error persisting incident: {e}", exc_info=True)

        return {
            "alert_state": alert_state,
            "alert_info": alert_info,
            "event_logged": should_persist,
            "event_id": event_id,
            "snapshot_path": snapshot_path
        }
