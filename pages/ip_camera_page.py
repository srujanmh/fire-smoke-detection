"""
IP Camera Stream Page.
Connects to external RTSP / HTTP / HTTPS video streams with
credential sanitization, real-time threat detection, and audio/visual alarms.
"""

import time
import streamlit as st
from typing import Dict, Any

from modules.ip_camera import IPCameraManager
from utils.image_utils import draw_detections
from utils.logging_config import sanitize_stream_url


def render_ip_camera(core: Dict[str, Any], ip_camera: IPCameraManager):
    detector = core["detector"]
    event_manager = core["events"]
    alerts = core["alerts"]
    config = core["config"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #10B981;">
            <h1>🌐 Network IP Camera Monitor</h1>
            <p>Connect to remote CCTV, IP cameras, or RTSP surveillance streams.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Supported protocols: `rtsp://`, `http://`, `https://`, `rtmp://`. "
        "Credentials in the URL are masked automatically for privacy."
    )

    col_url, col_conf = st.columns([3, 1])
    with col_url:
        stream_url = st.text_input(
            "Stream URL",
            value="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
            placeholder="rtsp://username:password@192.168.1.100:554/stream1"
        )
    with col_conf:
        conf_slider = st.slider("Confidence", min_value=0.10, max_value=0.95, value=float(config.confidence_threshold), step=0.05)

    btn_c1, btn_c2 = st.columns([1, 1])
    with btn_c1:
        connect_btn = st.button("🔗 Connect & Start Stream", type="primary", use_container_width=True)
    with btn_c2:
        disconnect_btn = st.button("⏹ Disconnect Stream", use_container_width=True)

    if disconnect_btn:
        ip_camera.disconnect()
        st.info("Stream disconnected.")

    if connect_btn:
        with st.spinner("Connecting to remote video stream..."):
            success, msg = ip_camera.connect(stream_url)
            if not success:
                st.error(f"Connection Failed: {msg}")

    view_col, tele_col = st.columns([3, 1.5])
    frame_placeholder = view_col.empty()
    status_placeholder = tele_col.empty()
    audio_placeholder = st.empty()

    if ip_camera.is_connected:
        st.caption(f"Connected to stream: `{ip_camera.masked_url}`")

        while ip_camera.is_connected:
            ret, frame_rgb = ip_camera.read_frame()
            if not ret or frame_rgb is None:
                st.warning("Lost connection or stream ended.")
                ip_camera.disconnect()
                break

            # YOLO detection
            det_result = detector.detect(frame_rgb, confidence_threshold=conf_slider)

            # Event / alert processing
            incident_outcome = event_manager.process_incident(
                frame=frame_rgb,
                detection_result=det_result,
                source="ip_camera"
            )
            alert_state = incident_outcome["alert_state"]
            alert_info = incident_outcome["alert_info"]

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

            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            status = ip_camera.get_status()
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
                    <h4 style="margin-top: 0; color: #F9FAFB;">IP Stream Telemetry</h4>
                    <div style="margin-bottom: 12px;">
                        <span class="{badge_class}">{alert_state}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #9CA3AF; line-height: 1.6;">
                        <strong>Stream FPS:</strong> {status['fps']}<br>
                        <strong>Frame Count:</strong> {status['frame_count']}<br>
                        <strong>Inference:</strong> {det_result.get('inference_time_ms', 0):.1f} ms<br>
                        <strong>Fire Window:</strong> {alert_info['fire_buffer_count']}/{alert_info['buffer_capacity']}<br>
                        <strong>Smoke Window:</strong> {alert_info['smoke_buffer_count']}/{alert_info['buffer_capacity']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(0.02)
    else:
        frame_placeholder.markdown(
            """
            <div style="background-color: #1F2937; height: 350px; display: flex; align-items: center; justify-content: center; border-radius: 8px; border: 2px dashed #374151; color: #9CA3AF;">
                <div>
                    <h3 style="margin-bottom: 5px;">IP Stream Inactive</h3>
                    <p>Enter RTSP / HTTP URL and click "Connect & Start Stream" above.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
