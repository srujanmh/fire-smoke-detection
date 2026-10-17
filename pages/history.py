"""
Detection History & Forensic Log Page.
Searchable, filterable audit trail of all detected incidents
with snapshot viewing, false-positive labeling, and CSV export.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_history(core: Dict[str, Any]):
    db = core["db"]
    config = core["config"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #F59E0B;">
            <h1>📜 Forensic Incident History</h1>
            <p>Cryptographically timestamped incident database with image proof and human-in-the-loop review.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 2, 2])
    with f_col1:
        class_filter = st.selectbox("Filter Class", ["All", "Fire", "Smoke", "Fire_and_Smoke"])
    with f_col2:
        source_filter = st.selectbox("Filter Source", ["All", "webcam", "image_upload", "video_upload", "ip_camera"])
    with f_col3:
        start_date = st.date_input("Start Date", value=None)
    with f_col4:
        end_date = st.date_input("End Date", value=None)

    # 2. Fetch Events
    events = db.get_events(
        limit=200,
        class_filter=class_filter if class_filter != "All" else None,
        source_filter=source_filter if source_filter != "All" else None,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )

    # 3. CSV Export
    csv_col1, csv_col2 = st.columns([3, 1])
    with csv_col2:
        export_path = os.path.join(config.reports_dir, "incident_history_export.csv")
        if st.button("📥 Export History to CSV", use_container_width=True):
            saved_file = db.export_to_csv(export_path)
            with open(saved_file, "rb") as cf:
                st.download_button(
                    label="⬇ Download CSV File",
                    data=cf.read(),
                    file_name="fire_smoke_incident_history.csv",
                    mime="text/csv"
                )

    st.markdown(f"**Found {len(events)} incident records:**")

    if not events:
        st.info("No detection events match the current filter criteria.")
        return

    # Convert to presentation table
    table_rows = []
    for ev in events:
        table_rows.append({
            "ID": ev["event_id"],
            "Timestamp": ev["timestamp"],
            "Source": ev["source"],
            "Class": ev["detected_class"].upper(),
            "Confidence": f"{ev['confidence']:.1%}",
            "Latency (ms)": f"{ev['processing_duration']:.1f}" if ev['processing_duration'] else "N/A",
            "Has Snapshot": "✅ Yes" if ev.get("snapshot_path") and os.path.exists(ev["snapshot_path"]) else "❌ No"
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # 4. Detailed Inspector & False Positive Review
    st.subheader("🔍 Incident Inspector & Quality Audit")
    event_ids = [ev["event_id"] for ev in events]
    selected_id = st.selectbox("Select Event ID to Inspect", event_ids)

    selected_event = next((ev for ev in events if ev["event_id"] == selected_id), None)
    if selected_event:
        d_col1, d_col2 = st.columns([1.5, 1.5])
        with d_col1:
            st.markdown(f"### Incident #{selected_event['event_id']}")
            st.write(f"**Timestamp:** {selected_event['timestamp']}")
            st.write(f"**Source:** {selected_event['source']}")
            st.write(f"**Detected Class:** {selected_event['detected_class'].upper()}")
            st.write(f"**Confidence:** {selected_event['confidence']:.2%}")
            st.write(f"**Alert Status:** {selected_event['alert_status']}")
            if selected_event.get("video_timestamp"):
                st.write(f"**Video Frame Time:** {selected_event['video_timestamp']:.2f} s")

            # Human Review Form
            st.markdown("#### Quality Audit (False-Positive Review)")
            rev_choice = st.radio(
                "Audit Label",
                ["Correct Detection", "False Positive", "Uncertain"],
                key=f"hist_rev_choice_{selected_id}"
            )
            rev_notes = st.text_input("Reviewer Notes", key=f"hist_notes_{selected_id}")
            if st.button("Save Audit Label", key=f"hist_btn_{selected_id}", type="primary"):
                status_code = rev_choice.lower().replace(" ", "_")
                db.insert_review(event_id=selected_id, reviewer_status=status_code, notes=rev_notes)
                st.success("Audit label recorded in SQLite database.")

        with d_col2:
            snap_path = selected_event.get("snapshot_path")
            if snap_path and os.path.exists(snap_path):
                st.markdown("#### Forensic Snapshot")
                st.image(snap_path, caption=f"Event #{selected_id} Stamped Image", use_container_width=True)
            else:
                st.info("No snapshot image file associated with this event.")
