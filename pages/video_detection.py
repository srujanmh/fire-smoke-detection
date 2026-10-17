"""
Video Detection Page.
Memory-efficient sequential video analysis with frame interval sampling,
progress monitoring, incident timestamp tracking, and annotated video export.
"""

import os
import tempfile
import streamlit as st
from typing import Dict, Any

from utils.file_validation import validate_video_file, sanitize_filename


def render_video_detection(core: Dict[str, Any]):
    video_proc = core["video_proc"]
    config = core["config"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #EC4899;">
            <h1>🎥 Video Stream Forensic Processor</h1>
            <p>Frame-by-frame temporal scanning of recorded footage with incident timestamp mapping.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        help=f"Maximum allowed file size: {config.max_upload_size_mb} MB"
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(config.confidence_threshold),
            step=0.05
        )
    with col_cfg2:
        frame_interval = st.select_slider(
            "Frame Processing Interval (Sampling Rate)",
            options=[1, 2, 3, 5, 10],
            value=config.video_process_frame_interval,
            help="1 = Analyze every frame (thorough, slower). 2 = Analyze every 2nd frame (2x faster)."
        )

    if uploaded_video is not None:
        clean_name = sanitize_filename(uploaded_video.name)

        # Save to temporary file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{clean_name}") as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_video_path = tmp_file.name

        try:
            # Metadata
            meta = video_proc.get_video_metadata(tmp_video_path)
            if meta.get("error"):
                st.error(f"Failed to read video metadata: {meta['error']}")
                return

            # Display video properties
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Resolution", f"{meta['width']}x{meta['height']}")
            col_m2.metric("Framerate", f"{meta['fps']} FPS")
            col_m3.metric("Duration", f"{meta['duration_sec']} s")
            col_m4.metric("Total Frames", f"{meta['total_frames']}")

            if st.button("🚀 Process & Annotate Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(pct: float, text: str):
                    progress_bar.progress(pct)
                    status_text.text(text)

                with st.spinner("Processing video frames..."):
                    result = video_proc.process_video(
                        input_video_path=tmp_video_path,
                        filename=clean_name,
                        frame_interval=frame_interval,
                        confidence_threshold=conf_thresh,
                        progress_callback=update_progress,
                        save_incident_snapshots=True
                    )

                if not result["success"]:
                    st.error(f"Video Processing Failed: {result.get('error')}")
                    return

                st.success("✅ Video processing and annotation complete!")

                # Summary Statistics
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Analyzed Frames", f"{result['analyzed_frames']}/{result['total_frames']}")
                s2.metric("Fire Incidents", result["fire_detections_count"])
                s3.metric("Smoke Incidents", result["smoke_detections_count"])
                s4.metric("Average Speed", f"{result['average_fps']} FPS")

                # Sample frame view
                if result.get("last_annotated_frame") is not None:
                    st.subheader("Sample Processed Frame")
                    st.image(result["last_annotated_frame"], use_container_width=True)

                # Download processed video
                out_path = result["output_video_path"]
                if os.path.exists(out_path):
                    with open(out_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.download_button(
                        label="⬇ Download Processed Video",
                        data=video_bytes,
                        file_name=os.path.basename(out_path),
                        mime="video/mp4",
                        type="primary"
                    )

                # Incident Timeline Table
                if result.get("sample_detections"):
                    st.subheader("⏱ Incident Timeline in Video")
                    timeline_data = []
                    for s in result["sample_detections"]:
                        for d in s["detections"]:
                            timeline_data.append({
                                "Video Timestamp (s)": s["timestamp"],
                                "Frame #": s["frame"],
                                "Detected Object": d["class_name"].upper(),
                                "Confidence": f"{d['confidence']:.1%}"
                            })
                    st.dataframe(timeline_data, use_container_width=True)

        finally:
            # Clean up temp file
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass
    else:
        st.info("👆 Upload an MP4, AVI, or MOV video file to begin automated frame-by-frame analysis.")
