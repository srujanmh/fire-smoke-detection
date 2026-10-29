"""
Pure OpenCV Direct HighGUI Application (main.py).
Zero Streamlit, zero browser, zero web server.
High-speed native OpenCV window with real-time HUD telemetry,
temporal alerts, and keyboard shortcuts.
"""

import sys
import time
import cv2
import argparse
import numpy as np
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_default_config
from utils.logging_config import logger
from utils.image_utils import draw_detections, rgb_to_bgr, bgr_to_rgb, save_incident_snapshot
from modules.detector import FireSmokeDetector
from modules.database import DatabaseManager
from modules.alerts import AlertSystem
from modules.event_manager import EventManager
from modules.camera import CameraManager


def run_opencv_sentinel(
    camera_index: int = 0,
    video_source: str = "",
    conf_thresh: float = 0.35,
    simulated: bool = False
):
    config = get_default_config()
    db = DatabaseManager(config.database_path)
    alerts = AlertSystem(
        cooldown_seconds=config.alert_cooldown_seconds,
        temporal_window_size=config.temporal_window_frames,
        temporal_min_detections=config.temporal_min_detections,
        audio_enabled=config.enable_audio_alerts
    )
    event_manager = EventManager(
        db_manager=db,
        alert_system=alerts,
        snapshots_dir=config.snapshots_dir,
        log_cooldown_seconds=config.alert_cooldown_seconds
    )
    detector = FireSmokeDetector(
        model_path=config.model_path,
        confidence_threshold=conf_thresh,
        iou_threshold=config.iou_threshold,
        imgsz=config.inference_img_size
    )

    if simulated or not detector.is_loaded:
        detector.is_simulation_mode = True
        logger.info("Running in SIMULATED demonstration mode.")

    # Open video feed
    if video_source:
        cap = cv2.VideoCapture(video_source)
        source_name = "video_file"
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        source_name = "webcam"

    if not cap.isOpened():
        print(f"Error: Unable to open video source (index: {camera_index}, file: '{video_source}').")
        return

    window_name = "AegisVision AI — OpenCV Native Sentinel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    print("\n" + "=" * 65)
    print(" 🔥 AegisVision AI — OpenCV Native Sentinel Active")
    print("=" * 65)
    print(" Keyboard Controls:")
    print("   [q] or [ESC] : Quit application")
    print("   [s]          : Manually capture forensic snapshot")
    print("   [m]          : Toggle Simulated Mode on/off")
    print("   [r]          : Reset temporal alert buffer")
    print("=" * 65 + "\n")

    fps_estimate = 0.0
    last_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                if video_source:
                    print("End of video file reached.")
                else:
                    print("Camera frame dropped or disconnected.")
                break

            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt) if fps_estimate > 0 else (1.0 / dt)
            last_time = now

            # RGB for detector
            frame_rgb = bgr_to_rgb(frame_bgr)

            # YOLO inference
            det_result = detector.detect(frame_rgb, confidence_threshold=conf_thresh, is_rgb=True)

            # Event / alert handling
            incident = event_manager.process_incident(frame_rgb, det_result, source=source_name)
            alert_state = incident["alert_state"]
            alert_info = incident["alert_info"]

            # Draw bounding boxes (BGR mode for OpenCV)
            annotated_bgr = draw_detections(frame_bgr, det_result.get("detections", []), is_rgb=False)

            # --- HUD OVERLAY ---
            # Top status banner
            h, w = annotated_bgr.shape[:2]
            cv2.rectangle(annotated_bgr, (0, 0), (w, 45), (20, 24, 31), -1)

            # Threat Badge Color
            if "FIRE" in alert_state:
                badge_bg = (68, 68, 239)   # Red BGR
            elif "SMOKE" in alert_state or alert_state == "UNCERTAIN":
                badge_bg = (11, 158, 245)  # Amber BGR
            else:
                badge_bg = (94, 197, 34)   # Green BGR

            cv2.rectangle(annotated_bgr, (15, 8), (240, 36), badge_bg, -1)
            cv2.putText(annotated_bgr, alert_state, (22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Telemetry text
            hud_text = f"FPS: {fps_estimate:.1f} | Latency: {det_result.get('inference_time_ms', 0):.1f}ms | FireBuf: {alert_info['fire_buffer_count']}/{alert_info['buffer_capacity']}"
            if det_result.get("is_simulated"):
                hud_text += " | [SIMULATED]"

            cv2.putText(annotated_bgr, hud_text, (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            cv2.imshow(window_name, annotated_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' or ESC
                break
            elif key == ord('s'):
                path = save_incident_snapshot(annotated_bgr, config.snapshots_dir, prefix="manual", is_rgb=False)
                print(f"Manual snapshot saved: {path}")
            elif key == ord('m'):
                detector.is_simulation_mode = not detector.is_simulation_mode
                print(f"Simulation Mode toggled: {detector.is_simulation_mode}")
            elif key == ord('r'):
                alerts.reset_state()
                print("Alert state and temporal buffer reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Surveillance stopped. Hardware released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisVision AI OpenCV Sentinel")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default="", help="Path to video file (optional)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--sim", action="store_true", help="Force simulated mode")
    args = parser.parse_args()

    run_opencv_sentinel(
        camera_index=args.camera,
        video_source=args.video,
        conf_thresh=args.conf,
        simulated=args.sim
    )
