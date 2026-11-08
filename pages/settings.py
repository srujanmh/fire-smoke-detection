"""
System Settings Page.
Allows interactive reconfiguration of model paths, inference devices,
detection thresholds, temporal window sizes, and simulated test mode.
"""

import os
import streamlit as st
from typing import Dict, Any

from utils.config import AppConfig


def render_settings(core: Dict[str, Any]):
    config: AppConfig = core["config"]
    detector = core["detector"]
    alerts = core["alerts"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #6B7280;">
            <h1>⚙️ System Configuration & Diagnostics</h1>
            <p>Fine-tune detection hyperparameters, inference devices, and persistence settings.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Model & Inference Hardware
    st.subheader("🤖 Neural Network & Inference Device")
    m_col1, m_col2 = st.columns([2.5, 1.5])
    with m_col1:
        new_model_path = st.text_input("YOLO Model Weights Path (.pt)", value=detector.model_path)
    with m_col2:
        hw = detector.get_hardware_info()
        device_opts = ["cpu"]
        if hw["cuda_available"]:
            device_opts.append("cuda")
        selected_device = st.selectbox("Inference Device", device_opts, index=0 if detector.device == "cpu" else 1)

    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        if st.button("🔄 Reload Model Weights", type="primary", use_container_width=True):
            detector.model_path = new_model_path
            detector.device = selected_device
            success, msg = detector.load_model()
            if success:
                st.success(f"Model reloaded successfully! Detected classes: {detector.class_mapping}")
            else:
                st.error(f"Model load failed: {msg}")

    with c_btn2:
        if st.button("⏹ Unload Model From Memory", use_container_width=True):
            detector.unload_model()
            st.info("Model unloaded.")

    # 2. Simulated Mode Toggle
    st.markdown("---")
    st.subheader("🧪 UI Demonstration & Testing Mode")
    sim_mode = st.checkbox(
        "Enable Simulated Mode (Non-AI Mock Inference)",
        value=detector.is_simulation_mode,
        help="Allows full end-to-end testing of UI widgets, audio alarms, database logging, and snapshots when weights are not present."
    )
    if sim_mode != detector.is_simulation_mode:
        detector.is_simulation_mode = sim_mode
        st.rerun()

    if detector.is_simulation_mode:
        st.warning("⚠️ SIMULATED MODE IS ACTIVE. Detections are mock overlays for UI verification, NOT real AI inference.")

    # 3. Detection Thresholds
    st.markdown("---")
    st.subheader("🎯 Detection & Alarm Thresholds")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        new_fire_thresh = st.slider("Fire Threshold", 0.10, 0.95, float(detector.fire_threshold), 0.05)
        detector.fire_threshold = new_fire_thresh
    with t_col2:
        new_smoke_thresh = st.slider("Smoke Threshold", 0.10, 0.95, float(detector.smoke_threshold), 0.05)
        detector.smoke_threshold = new_smoke_thresh
    with t_col3:
        new_iou_thresh = st.slider("IoU NMS Threshold", 0.10, 0.90, float(detector.iou_threshold), 0.05)
        detector.iou_threshold = new_iou_thresh

    # 4. Temporal Confirmation & Audio
    st.markdown("---")
    st.subheader("⏱️ Temporal Confirmation & Audio Sentinel")
    a_col1, a_col2, a_col3 = st.columns(3)
    with a_col1:
        window_size = st.number_input("Sliding Window Frames", min_value=1, max_value=20, value=alerts.temporal_window_size, step=1)
    with a_col2:
        min_detections = st.number_input("Min Confirmed Detections", min_value=1, max_value=int(window_size), value=min(alerts.temporal_min_detections, int(window_size)), step=1)
    # update_temporal_window() rebuilds the deques with the correct maxlen —
    # direct attribute assignment leaves the deque's immutable maxlen unchanged.
    alerts.update_temporal_window(int(window_size), int(min_detections))
    with a_col3:
        cooldown_sec = st.number_input("Audio Alarm Cooldown (s)", min_value=1.0, max_value=60.0, value=float(alerts.cooldown_seconds), step=1.0)
        alerts.cooldown_seconds = float(cooldown_sec)

    # 5. Storage Directories
    st.markdown("---")
    st.subheader("💾 Storage & Directory Paths")
    st.text_input("Database Path", value=config.database_path, disabled=True)
    st.text_input("Snapshots Directory", value=config.snapshots_dir, disabled=True)
    st.text_input("Processed Videos Directory", value=config.processed_videos_dir, disabled=True)
