"""
Analytics & Trends Page.
Visual intelligence on detection distributions, confidence profiles,
and False Positive audit telemetry.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any


def render_analytics(core: Dict[str, Any]):
    db = core["db"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #3B82F6;">
            <h1>📈 Detection Intelligence & Analytics</h1>
            <p>Empirical trend analysis, confidence distributions, and false-positive audit findings.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_events = db.get_events_dataframe(limit=5000)
    reviews_summary = db.get_reviews_summary()

    if df_events.empty:
        st.info("No detection events available for analytics. Run detections to populate charts.")
        return

    # 1. Top Level Metrics
    total_records = len(df_events)
    avg_conf = df_events["confidence"].mean() if "confidence" in df_events.columns else 0.0
    avg_latency = df_events["processing_duration"].mean() if "processing_duration" in df_events.columns else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Analyzed Incidents", total_records)
    m2.metric("Mean Detection Confidence", f"{avg_conf:.1%}")
    m3.metric("Mean Latency", f"{avg_latency:.1f} ms")
    m4.metric("Human Audited Events", sum(reviews_summary.values()))

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 Threat Class Breakdown")
        cls_counts = df_events["detected_class"].value_counts().reset_index()
        cls_counts.columns = ["class", "count"]
        fig_donut = px.pie(
            cls_counts,
            names="class",
            values="count",
            hole=0.4,
            color="class",
            color_discrete_map={"fire": "#EF4444", "smoke": "#F59E0B", "fire_and_smoke": "#EC4899"}
        )
        fig_donut.update_layout(
            paper_bgcolor="#1F2937",
            plot_bgcolor="#1F2937",
            font_color="#F9FAFB",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.subheader("🛡️ False-Positive Audit Results")
        if reviews_summary:
            rev_df = pd.DataFrame(list(reviews_summary.items()), columns=["Status", "Count"])
            rev_df["Status"] = rev_df["Status"].str.replace("_", " ").str.title()
            fig_rev = px.bar(
                rev_df,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map={"Correct Detection": "#22C55E", "False Positive": "#EF4444", "Uncertain": "#F59E0B"}
            )
            fig_rev.update_layout(
                paper_bgcolor="#1F2937",
                plot_bgcolor="#1F2937",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No false-positive reviews submitted yet. Audit events in 'Detection History' or 'Image Detection'.")

    # 3. Known Confounders & False-Positive Risk Factors Guide
    st.markdown("---")
    st.subheader("⚠️ Common Optical Confounders in Computer Vision")
    st.markdown(
        """
        Final-year AI/ML projects must explain why real-world optical detection faces false positives.
        The table below details known environmental confounders and how they challenge standard vision models:
        """
    )

    confounders = [
        {"Confounder": "Vehicle Headlights & Streetlamps", "Target Confused": "Fire / Flame", "Mechanism": "High localized pixel luminance, bright glare halo mimicking flame core."},
        {"Confounder": "Sunsets & Golden Hour Sunlight", "Target Confused": "Fire / Flame", "Mechanism": "Diffuse orange/red wavelength reflection on glass windows or clouds."},
        {"Confounder": "Orange & Hi-Vis Safety Clothing", "Target Confused": "Fire", "Mechanism": "High color saturation in the 590-620nm spectrum; requires spatial shape reasoning."},
        {"Confounder": "Boiler Steam & Cooling Vapor", "Target Confused": "Smoke", "Mechanism": "White, semi-translucent particulate dispersion similar to early stage smoke."},
        {"Confounder": "Atmospheric Fog & Industrial Dust", "Target Confused": "Smoke", "Mechanism": "Lowers contrast, produces volumetric hazing across background pixels."},
        {"Confounder": "Kitchen Cooking Steam & Sautéing", "Target Confused": "Fire & Smoke", "Mechanism": "Intermittent hot vapor bursts and stove burner illumination."}
    ]
    st.table(pd.DataFrame(confounders))
