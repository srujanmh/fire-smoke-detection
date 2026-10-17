"""
Image Detection Page.
Upload and analyze static imagery with YOLO inference,
side-by-side visualization, detection metrics, and false-positive audit review.
"""

import streamlit as st
from typing import Dict, Any

from utils.image_utils import get_image_base64


def render_image_detection(core: Dict[str, Any]):
    image_proc = core["image_proc"]
    config = core["config"]
    db = core["db"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #8B5CF6;">
            <h1>🖼️ Static Image Threat Analysis</h1>
            <p>Upload forensic snapshots or scene photographs for comprehensive object-level inspection.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help=f"Maximum allowed file size: {config.max_upload_size_mb} MB"
        )
    with c2:
        conf_thresh = st.slider(
            "Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(config.confidence_threshold),
            step=0.05
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        with st.spinner("Analyzing image through YOLO neural network..."):
            result = image_proc.process_image(
                file_bytes=file_bytes,
                filename=filename,
                confidence_threshold=conf_thresh,
                save_incident=True
            )

        if not result["success"]:
            st.error(f"Image Analysis Failed: {result.get('error')}")
            return

        summary = result["summary"]
        status = result["status"]

        # Alert banner
        badge_class = "badge-normal"
        if "FIRE" in status:
            badge_class = "badge-fire"
        elif "SMOKE" in status:
            badge_class = "badge-smoke"
        elif status == "UNCERTAIN":
            badge_class = "badge-smoke"

        st.markdown(
            f"""
            <div style="background-color: #1F2937; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid {'#EF4444' if 'FIRE' in status else ('#F59E0B' if 'SMOKE' in status else '#22C55E')};">
                <span class="{badge_class}" style="margin-right: 10px;">{status}</span>
                <span style="color: #F9FAFB; font-weight: 500;">{result['status_message']}</span>
                {f'<span class="badge-simulated" style="margin-left: 10px;">[SIMULATED - NON-AI]</span>' if summary.get('is_simulated') else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Side by Side comparison
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Original Image")
            st.image(result["original_image"], use_container_width=True)
        with col_img2:
            st.subheader("Annotated Detections")
            st.image(result["annotated_image"], use_container_width=True)

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fire Objects", summary["fire_count"])
        m2.metric("Smoke Objects", summary["smoke_count"])
        m3.metric("Peak Confidence", f"{summary['highest_confidence']:.1%}")
        m4.metric("Inference Time", f"{summary['processing_time_ms']:.1f} ms")

        # Download Button
        st.download_button(
            label="⬇ Download Annotated Image",
            data=result["download_bytes"],
            file_name=f"annotated_{filename}",
            mime="image/jpeg",
            type="primary"
        )

        # Optional Forensic Audit Feedback (False-Positive Reporting)
        incident_info = result.get("incident_info")
        if incident_info and incident_info.get("event_id"):
            event_id = incident_info["event_id"]
            st.markdown("---")
            st.subheader("🔍 Quality Assurance Audit (False Positive Review)")
            st.caption("Help refine model validation by reviewing this recorded incident.")

            col_rev1, col_rev2 = st.columns([1, 2])
            with col_rev1:
                review_choice = st.radio(
                    "Audit Classification",
                    ["Correct Detection", "False Positive", "Uncertain"],
                    key=f"rev_choice_{event_id}"
                )
            with col_rev2:
                review_notes = st.text_input("Reviewer Notes (e.g., sunset reflection, steam from cooking)", key=f"rev_notes_{event_id}")
                if st.button("Submit Audit Label", key=f"rev_btn_{event_id}"):
                    status_code = review_choice.lower().replace(" ", "_")
                    db.insert_review(event_id=event_id, reviewer_status=status_code, notes=review_notes)
                    st.success("Audit feedback saved to database!")
    else:
        st.info("👆 Upload an image above to start automated fire and smoke detection.")
