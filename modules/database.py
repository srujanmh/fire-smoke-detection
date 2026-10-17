"""
Database Management Module for Fire & Smoke Detection.
Uses SQLite with strictly parameterized queries to store detection events,
incident snapshots, false-positive reviews, and query summaries.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from utils.logging_config import logger


class DatabaseManager:
    """Thread-safe SQLite database manager for detection events and reviews."""

    def __init__(self, db_path: str = "database/detection_events.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self.init_database()

    def _ensure_db_dir(self) -> None:
        """Create parent directory if it does not exist."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Return a configured connection with row_factory set to sqlite3.Row."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_database(self) -> None:
        """Create tables and indexes if they do not exist."""
        create_events_table = """
        CREATE TABLE IF NOT EXISTS detection_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            filename TEXT,
            detected_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_json TEXT,
            video_timestamp REAL,
            alert_status TEXT NOT NULL,
            snapshot_path TEXT,
            processing_duration REAL
        );
        """

        create_reviews_table = """
        CREATE TABLE IF NOT EXISTS incident_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            reviewer_status TEXT NOT NULL,
            notes TEXT,
            review_timestamp TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES detection_events (event_id) ON DELETE CASCADE
        );
        """

        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON detection_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_class ON detection_events(detected_class);
        CREATE INDEX IF NOT EXISTS idx_events_source ON detection_events(source);
        CREATE INDEX IF NOT EXISTS idx_reviews_event ON incident_reviews(event_id);
        """

        try:
            with self.get_connection() as conn:
                conn.execute(create_events_table)
                conn.execute(create_reviews_table)
                conn.executescript(create_indexes)
                conn.commit()
            logger.info(f"Database initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise

    def insert_detection_event(
        self,
        source: str,
        detected_class: str,
        confidence: float,
        alert_status: str,
        filename: Optional[str] = None,
        bbox_data: Optional[List[Dict[str, Any]]] = None,
        video_timestamp: Optional[float] = None,
        snapshot_path: Optional[str] = None,
        processing_duration: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Insert a new detection incident record using parameterized queries.
        Returns the inserted event_id.
        """
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox_json_str = json.dumps(bbox_data) if bbox_data else None

        sql = """
        INSERT INTO detection_events (
            timestamp, source, filename, detected_class, confidence,
            bbox_json, video_timestamp, alert_status, snapshot_path, processing_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (
                    ts,
                    source,
                    filename,
                    detected_class,
                    float(confidence),
                    bbox_json_str,
                    video_timestamp,
                    alert_status,
                    snapshot_path,
                    processing_duration
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        class_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve filtered, paginated events with optional class, source, and date filtering.
        """
        query = "SELECT * FROM detection_events WHERE 1=1"
        params: List[Any] = []

        if class_filter and class_filter.lower() != "all":
            query += " AND LOWER(detected_class) LIKE ?"
            params.append(f"%{class_filter.lower()}%")

        if source_filter and source_filter.lower() != "all":
            query += " AND LOWER(source) = ?"
            params.append(source_filter.lower())

        if start_date:
            query += " AND timestamp >= ?"
            params.append(f"{start_date} 00:00:00")

        if end_date:
            query += " AND timestamp <= ?"
            params.append(f"{end_date} 23:59:59")

        query += " ORDER BY event_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute aggregated statistics for the dashboard overview."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total events
            cursor.execute("SELECT COUNT(*) FROM detection_events;")
            total_events = cursor.fetchone()[0]

            # Fire events
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE LOWER(detected_class) LIKE '%fire%';")
            fire_events = cursor.fetchone()[0]

            # Smoke events
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE LOWER(detected_class) LIKE '%smoke%';")
            smoke_events = cursor.fetchone()[0]

            # Unique images analyzed
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM detection_events WHERE source = 'image_upload';")
            images_analyzed = cursor.fetchone()[0]

            # Unique videos analyzed
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM detection_events WHERE source = 'video_upload';")
            videos_analyzed = cursor.fetchone()[0]

            # Webcam sessions/events
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE source = 'webcam';")
            webcam_events = cursor.fetchone()[0]

            return {
                "total_events": total_events,
                "fire_events": fire_events,
                "smoke_events": smoke_events,
                "images_analyzed": images_analyzed,
                "videos_analyzed": videos_analyzed,
                "webcam_events": webcam_events
            }

    def get_events_dataframe(
        self,
        limit: int = 1000,
        class_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch events directly into a Pandas DataFrame for analytics and charts."""
        query = "SELECT * FROM detection_events WHERE 1=1"
        params: List[Any] = []

        if class_filter and class_filter.lower() != "all":
            query += " AND LOWER(detected_class) LIKE ?"
            params.append(f"%{class_filter.lower()}%")

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df

    def export_to_csv(self, output_path: str) -> str:
        """Export all detection events to a CSV file."""
        df = self.get_events_dataframe(limit=50000)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} detection events to: {output_path}")
        return output_path

    def insert_review(
        self,
        event_id: int,
        reviewer_status: str,
        notes: Optional[str] = None
    ) -> int:
        """Record human audit review (correct, false_positive, uncertain) for an incident."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = """
        INSERT INTO incident_reviews (event_id, reviewer_status, notes, review_timestamp)
        VALUES (?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (event_id, reviewer_status.lower(), notes, now_str))
            conn.commit()
            return cursor.lastrowid

    def get_reviews_summary(self) -> Dict[str, int]:
        """Return counts of audited false positives vs. true positives."""
        sql = """
        SELECT reviewer_status, COUNT(*) as count
        FROM incident_reviews
        GROUP BY reviewer_status;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return {row["reviewer_status"]: row["count"] for row in rows}
