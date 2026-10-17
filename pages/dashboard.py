"""
Dashboard Overview Page for Fire & Smoke Detection System.
Renders summary metrics, Plotly interactive analytics,
and recent incident snapshot gallery.
"""

import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any


def render_dashboard(core: Dict[str, Any]):
    db = core["db"]
    detector = core["detector"]

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🔥 System Operations Center</h1>
            <p>Real-time telemetry, threat detection overview, and forensic activity log.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Fetch persistent summary statistics
    metrics = db.get_summary_metrics()
    hw = detector.get_hardware_info()

    # Metric Cards Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #60A5FA;">{metrics['total_events']}</div>
                <div class="metric-label">Total Events</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #EF4444;">{metrics['fire_events']}</div>
                <div class="metric-label">Fire Alerts</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #F59E0B;">{metrics['smoke_events']}</div>
                <div class="metric-label">Smoke Alerts</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #A78BFA;">{metrics['images_analyzed']}</div>
                <div class="metric-label">Images Analyzed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #38BDF8;">{metrics['videos_analyzed']}</div>
                <div class="metric-label">Videos Processed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c6:
        status_text = "READY" if detector.is_loaded else ("SIMULATED" if detector.is_simulation_mode else "OFFLINE")
        status_color = "#22C55E" if detector.is_loaded else ("#818CF8" if detector.is_simulation_mode else "#EF4444")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {status_color}; font-size: 1.5rem; line-height: 2.3rem;">{status_text}</div>
                <div class="metric-label">Model ({hw['active_device'].upper()})</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Section
    chart_col1, chart_col2 = st.columns(2)

    df_events = db.get_events_dataframe(limit=1000)

    with chart_col1:
        st.subheader("📊 Fire vs. Smoke Incidents")
        if metrics['fire_events'] > 0 or metrics['smoke_events'] > 0:
            fig_pie = px.pie(
                names=["Fire Incidents", "Smoke Incidents"],
                values=[metrics['fire_events'], metrics['smoke_events']],
                color=["Fire Incidents", "Smoke Incidents"],
                color_discrete_map={"Fire Incidents": "#EF4444", "Smoke Incidents": "#F59E0B"},
                hole=0.45
            )
            fig_pie.update_layout(
                paper_bgcolor="#1F2937",
                plot_bgcolor="#1F2937",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No detection events recorded in database yet. Run detections to populate charts.")

    with chart_col2:
        st.subheader("📡 Incidents by Source")
        if not df_events.empty and "source" in df_events.columns:
            source_counts = df_events["source"].value_counts().reset_index()
            source_counts.columns = ["source", "count"]
            fig_bar = px.bar(
                source_counts,
                x="source",
                y="count",
                color="source",
                color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6", "#EC4899"],
                labels={"source": "Input Source", "count": "Events"}
            )
            fig_bar.update_layout(
                paper_bgcolor="#1F2937",
                plot_bgcolor="#1F2937",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No source activity recorded yet.")

    # 3. Timeline & Confidence Distribution
    if not df_events.empty:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("📈 Detection Timeline")
            df_events["dt"] = pd.to_datetime(df_events["timestamp"], errors="coerce")
            df_timeline = df_events.dropna(subset=["dt"]).set_index("dt").resample("1h").size().reset_index()
            df_timeline.columns = ["time", "event_count"]

            fig_line = px.line(
                df_timeline,
                x="time",
                y="event_count",
                labels={"time": "Time", "event_count": "Events / Hour"},
                color_discrete_sequence=["#EF4444"]
            )
            fig_line.update_layout(
                paper_bgcolor="#1F2937",
                plot_bgcolor="#1F2937",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col_t2:
            st.subheader("🎯 Confidence Score Distribution")
            fig_hist = px.histogram(
                df_events,
                x="confidence",
                nbins=20,
                color="detected_class",
                color_discrete_map={"fire": "#EF4444", "smoke": "#F59E0B", "fire_and_smoke": "#EC4899"},
                labels={"confidence": "Confidence Score"}
            )
            fig_hist.update_layout(
                paper_bgcolor="#1F2937",
                plot_bgcolor="#1F2937",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                barmode="overlay"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # 4. Recent Forensic Incident Snapshots Gallery
    st.subheader("📸 Recent Incident Snapshots")
    recent_events = db.get_events(limit=6)
    valid_snapshots = [e for e in recent_events if e.get("snapshot_path") and os.path.exists(e["snapshot_path"])]

    if valid_snapshots:
        cols = st.columns(3)
        for i, ev in enumerate(valid_snapshots):
            with cols[i % 3]:
                badge_color = "#EF4444" if "fire" in ev["detected_class"] else "#F59E0B"
                st.image(
                    ev["snapshot_path"],
                    caption=f"{ev['detected_class'].upper()} ({ev['confidence']:.1%}) - {ev['timestamp']}",
                    use_container_width=True
                )
    else:
        st.markdown(
            """
            <div style="background-color: #1F2937; padding: 25px; border-radius: 8px; text-align: center; color: #9CA3AF;">
                No incident snapshots saved yet. When confirmed fire or smoke is detected, high-resolution forensic snapshots are stamped here.
            </div>
            """,
            unsafe_allow_html=True
        )
