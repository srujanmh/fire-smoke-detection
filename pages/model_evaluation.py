"""
Model Evaluation Page.
Performs ground-truth empirical evaluation using IoU matching
against labeled YOLO test datasets. Calculates Precision, Recall, F1, and mAP@0.5.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_model_evaluation(core: Dict[str, Any]):
    evaluator = core["evaluator"]
    detector = core["detector"]
    config = core["config"]

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #6366F1;">
            <h1>🎯 Empirical Model Evaluation</h1>
            <p>Rigorous benchmark against labeled ground-truth datasets using spatial IoU matching.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not detector.is_loaded:
        st.warning(
            "⚠️ No active AI model is loaded. Evaluation requires a trained model weights file (.pt). "
            "Please check the Settings page or models/ README."
        )

    st.markdown(
        """
        ### Academic Rigor & Methodology
        Standard image classification metrics (like simple top-1 accuracy) are **invalid** for object detection.
        This evaluation engine performs **spatial Intersection over Union (IoU)** matching between predicted bounding boxes
        and ground-truth coordinates:
        - **True Positive (TP):** Predicted box overlaps a ground-truth box with $\text{IoU} \ge 0.50$ and matches the class.
        - **False Positive (FP):** Predicted box has no overlapping ground-truth box $\ge 0.50$ (spurious detection).
        - **False Negative (FN):** Ground-truth box that was completely missed by the detector.
        """
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        yaml_path = st.text_input("Dataset YAML Path", value=str(os.path.join(config.data_dir, "data.yaml")))
    with c2:
        iou_thresh = st.slider("IoU Threshold", min_value=0.30, max_value=0.80, value=0.50, step=0.05)
    with c3:
        conf_eval = st.slider("Eval Confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)

    if st.button("🚀 Run Empirical Evaluation", type="primary"):
        with st.spinner("Evaluating model on test dataset..."):
            res = evaluator.evaluate_test_dataset(
                data_yaml_path=yaml_path,
                iou_threshold=iou_thresh,
                conf_threshold=conf_eval
            )

        if not res["success"]:
            st.error(f"Evaluation Could Not Be Completed: {res.get('error')}")
            st.markdown(
                """
                #### How to set up the Test Dataset:
                1. Place your test images in `data/images/test/`.
                2. Place corresponding YOLO normalized label files in `data/labels/test/`.
                3. Ensure `data/data.yaml` points to `images/test`.
                4. Refer to `data/README.md` for exact label format and specifications.
                """
            )
            return

        st.success(f"✅ Evaluated {res['images_evaluated']} test images successfully!")

        # High level metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Precision", f"{res['overall_precision']:.1%}")
        m2.metric("Overall Recall", f"{res['overall_recall']:.1%}")
        m3.metric("F1-Score", f"{res['overall_f1']:.1%}")
        m4.metric("Mean AP@0.5", f"{res['map_50']:.1%}")

        # Per-class table
        st.subheader("Class-Level Detection Performance")
        pcm = res["per_class_metrics"]
        class_table = []
        for cls_name, vals in pcm.items():
            class_table.append({
                "Class": cls_name.upper(),
                "Ground Truth Boxes": vals["ground_truth_count"],
                "Predictions": vals["prediction_count"],
                "True Positives (TP)": vals["tp"],
                "False Positives (FP)": vals["fp"],
                "False Negatives (FN)": vals["fn"],
                "Precision": f"{vals['precision']:.1%}",
                "Recall": f"{vals['recall']:.1%}",
                "F1-Score": f"{vals['f1_score']:.1%}"
            })
        st.dataframe(pd.DataFrame(class_table), use_container_width=True)

        # Latency & Throughput
        st.subheader("Inference Throughput Benchmark")
        l1, l2 = st.columns(2)
        l1.metric("Mean Frame Latency", f"{res['avg_latency_ms']:.1f} ms")
        l2.metric("Inference Throughput", f"{res['throughput_fps']} FPS")
