"""
Live Webcam Detection Page.
Interactive real-time camera stream with YOLO bounding boxes,
temporal alert confirmation, audio alarms, latency/FPS counters,
and forensic incident recording.
"""

import time
import streamlit as st
import numpy as np
from typing import Dict, Any

from modules.camera import CameraManager
from utils.image_utils import draw_detections, save_incident_snapshot
from utils.logging_config import logger


def render_live_camera(core: Dict[str, Any], camera: CameraManager):
    detector = core["detector"]
    event_manager = core["events"]
    alerts = core["alerts"]
    config = core["config"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #3B82F6;">
            <h1>📹 Real-Time Live Camera Sentinel</h1>
            <p>Direct optical surveillance with sub-second YOLO inference and temporal threat confirmation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Device and Model Readiness Checks
    if not detector.is_loaded and not detector.is_simulation_mode:
        st.warning(
            "⚠️ No fire/smoke detection model is currently loaded. "
            "Please ensure weights exist at `models/fire_smoke.pt` or activate **Simulated Mode** in Settings."
        )

    # Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 2, 2])
    with ctrl_col1:
        cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=config.default_camera_index, step=1)
    with ctrl_col2:
        conf_slider = st.slider("Confidence", min_value=0.10, max_value=0.95, value=float(config.confidence_threshold), step=0.05)
    with ctrl_col3:
        enable_audio = st.checkbox("Audio Alerts", value=config.enable_audio_alerts)
        alerts.audio_enabled = enable_audio
    with ctrl_col4:
        silence_audio = st.checkbox("Mute Audio", value=alerts.silenced)
        alerts.set_silenced(silence_audio)

    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])
    with btn_col1:
        start_btn = st.button("▶ Start Surveillance", type="primary", use_container_width=True)
    with btn_col2:
        stop_btn = st.button("⏹ Stop Camera", use_container_width=True)
    with btn_col3:
        reset_alert_btn = st.button("🔄 Reset Alert Buffer", use_container_width=True)
        if reset_alert_btn:
            alerts.reset_state()
            st.success("Alert buffer cleared.")

    # Camera state handling
    if stop_btn:
        camera.stop()
        st.info("Camera stopped.")

    if start_btn:
        success, msg = camera.start(camera_index=int(cam_index))
        if not success:
            st.error(f"Failed to start camera: {msg}")

    # Layout for Video Display and Telemetry Panel
    view_col, tele_col = st.columns([3, 1.5])
    frame_placeholder = view_col.empty()
    status_placeholder = tele_col.empty()
    audio_placeholder = st.empty()

    if camera.is_active:
        st.caption("🟢 Camera surveillance loop active. Press 'Stop Camera' to release hardware.")

        # Streaming loop
        # Note: In Streamlit, a controlled while loop handles frame updates smoothly
        while camera.is_active:
            ret, frame_rgb = camera.read_frame()
            if not ret or frame_rgb is None:
                st.error("Lost video feed or failed to grab frame.")
                camera.stop()
                break

            # Run YOLO Detection
            t_start = time.time()
            det_result = detector.detect(frame_rgb, confidence_threshold=conf_slider)
            inference_ms = det_result.get("inference_time_ms", 0.0)

            # Process through Event & Alert Manager (Temporal confirmation)
            incident_outcome = event_manager.process_incident(
                frame=frame_rgb,
                detection_result=det_result,
                source="webcam"
            )
            alert_state = incident_outcome["alert_state"]
            alert_info = incident_outcome["alert_info"]

            # Draw annotations on frame
            annotated_frame = draw_detections(
                frame_rgb,
                det_result.get("detections", []),
                is_rgb=True
            )

            # Web Audio Trigger
            if alert_info.get("should_play_sound", False):
                audio_placeholder.markdown(
                    alerts.get_web_audio_html(alert_state),
                    unsafe_allow_html=True
                )

            # Render video frame
            frame_placeholder.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

            # Render Telemetry Box
            stats = camera.get_stats()
            badge_class = "badge-normal"
            if "FIRE" in alert_state:
                badge_class = "badge-fire"
            elif "SMOKE" in alert_state:
                badge_class = "badge-smoke"
            elif alert_state == "UNCERTAIN":
                badge_class = "badge-smoke"

            status_placeholder.markdown(
                f"""
                <div style="background-color: #1F2937; padding: 15px; border-radius: 8px; border: 1px solid #374151;">
                    <h4 style="margin-top: 0; color: #F9FAFB;">Surveillance Status</h4>
                    <div style="margin-bottom: 12px;">
                        <span class="{badge_class}">{alert_state}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #9CA3AF; line-height: 1.6;">
                        <strong>FPS:</strong> {stats['fps']}<br>
                        <strong>Inference Latency:</strong> {inference_ms:.1f} ms<br>
                        <strong>Frame Count:</strong> {stats['frame_count']}<br>
                        <strong>Fire Buffer:</strong> {alert_info['fire_buffer_count']}/{alert_info['buffer_capacity']}<br>
                        <strong>Smoke Buffer:</strong> {alert_info['smoke_buffer_count']}/{alert_info['buffer_capacity']}<br>
                        <strong>Active Objects:</strong> {det_result.get('total_count', 0)}
                    </div>
                    {f'<div class="badge-simulated" style="margin-top: 10px;">[SIMULATED - NON-AI]</div>' if det_result.get('is_simulated') else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(0.01)  # Yield CPU cycles
    else:
        frame_placeholder.markdown(
            """
            <div style="background-color: #1F2937; height: 380px; display: flex; align-items: center; justify-content: center; border-radius: 8px; border: 2px dashed #374151; color: #9CA3AF;">
                <div>
                    <h3 style="margin-bottom: 5px;">Camera Standby</h3>
                    <p>Select camera index and click "Start Surveillance" above.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
