"""
Centralized Configuration Module for Fire & Smoke Detection System.
Provides strongly-typed settings, environment variable overrides,
and validation logic.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    """Master application configuration with defensive validation."""

    app_name: str = "AI Real-Time Fire & Smoke Detection System"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model Settings
    model_path: str = str(PROJECT_ROOT / "models" / "fire_smoke.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    inference_img_size: int = 640
    force_cpu: bool = False

    # Detection & Temporal Confirmation Settings
    fire_confidence_threshold: float = 0.35
    smoke_confidence_threshold: float = 0.30
    temporal_window_frames: int = 5
    temporal_min_detections: int = 3
    alert_cooldown_seconds: float = 5.0
    enable_audio_alerts: bool = True

    # Camera & Video Processing Settings
    default_camera_index: int = 0
    video_process_frame_interval: int = 2
    max_upload_size_mb: int = 150

    # Paths
    database_path: str = str(PROJECT_ROOT / "database" / "detection_events.db")
    snapshots_dir: str = str(PROJECT_ROOT / "outputs" / "snapshots")
    processed_videos_dir: str = str(PROJECT_ROOT / "outputs" / "processed_videos")
    reports_dir: str = str(PROJECT_ROOT / "outputs" / "reports")
    data_dir: str = str(PROJECT_ROOT / "data")

    # Simulation Mode (Allows full UI testing if no trained weights exist)
    simulation_mode: bool = False

    # UI Theme Palette (Dark industrial safety-monitoring theme)
    color_bg_dark: str = "#111827"
    color_card_bg: str = "#1F2937"
    color_fire: str = "#EF4444"
    color_smoke: str = "#F59E0B"
    color_safe: str = "#22C55E"
    color_text_primary: str = "#F9FAFB"
    color_text_secondary: str = "#9CA3AF"

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration parameters for consistency and safety."""
        if not (0.01 <= self.confidence_threshold <= 0.99):
            return False, "Confidence threshold must be between 0.01 and 0.99"
        if not (0.01 <= self.fire_confidence_threshold <= 0.99):
            return False, "Fire confidence threshold must be between 0.01 and 0.99"
        if not (0.01 <= self.smoke_confidence_threshold <= 0.99):
            return False, "Smoke confidence threshold must be between 0.01 and 0.99"
        if self.inference_img_size not in [320, 416, 512, 640, 1280]:
            return False, "Inference image size must be standard (320, 416, 512, 640, 1280)"
        if self.video_process_frame_interval < 1:
            return False, "Video processing interval must be >= 1"
        if self.max_upload_size_mb < 1:
            return False, "Max upload size must be positive"
        if self.temporal_window_frames < 1:
            return False, "Temporal window must be >= 1"
        if self.temporal_min_detections > self.temporal_window_frames:
            return False, "Temporal min detections cannot exceed window size"
        return True, "Configuration is valid"

    def ensure_directories(self) -> None:
        """Ensure all required output and database directories exist."""
        for path_str in [
            self.snapshots_dir,
            self.processed_videos_dir,
            self.reports_dir,
            Path(self.database_path).parent,
            self.data_dir,
            PROJECT_ROOT / "models"
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


def get_default_config() -> AppConfig:
    """Instantiate and return configuration initialized with environment overrides."""
    config = AppConfig()

    # Read environment variables if available
    if os.getenv("MODEL_PATH"):
        config.model_path = os.getenv("MODEL_PATH")
    if os.getenv("CONFIDENCE_THRESHOLD"):
        try:
            config.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
        except ValueError:
            pass
    if os.getenv("ALERT_COOLDOWN_SECONDS"):
        try:
            config.alert_cooldown_seconds = float(os.getenv("ALERT_COOLDOWN_SECONDS"))
        except ValueError:
            pass
    if os.getenv("DATABASE_PATH"):
        config.database_path = os.getenv("DATABASE_PATH")

    config.ensure_directories()
    return config
