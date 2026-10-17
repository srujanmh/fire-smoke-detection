"""
Dataset Preparation & YOLO Model Training Page.
Provides automated dataset integrity validation, class distribution analysis,
and supervised YOLO training configuration.
"""

import os
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


def validate_yolo_dataset(data_dir: Path) -> Dict[str, Any]:
    """
    Scans dataset directories for image/label integrity, orphaned files,
    invalid box bounds, and class distribution counts.
    """
    results = {
        "is_valid": True,
        "splits": {},
        "class_counts": {"fire": 0, "smoke": 0, "other": 0},
        "errors": [],
        "warnings": []
    }

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        img_count = 0
        lbl_count = 0
        orphaned_images = 0
        orphaned_labels = 0
        invalid_boxes = 0

        if img_dir.exists():
            images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in supported_exts}
            img_count = len(images)
        else:
            images = {}

        if lbl_dir.exists():
            labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"}
            lbl_count = len(labels)
        else:
            labels = {}

        # Check pairing
        for stem in images:
            if stem not in labels:
                orphaned_images += 1
            else:
                # Inspect boxes
                with open(labels[stem], "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                xc, yc, w, h = [float(x) for x in parts[1:5]]
                                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                                    invalid_boxes += 1
                                if cls_id == 0:
                                    results["class_counts"]["fire"] += 1
                                elif cls_id == 1:
                                    results["class_counts"]["smoke"] += 1
                                else:
                                    results["class_counts"]["other"] += 1
                            except ValueError:
                                invalid_boxes += 1

        for stem in labels:
            if stem not in images:
                orphaned_labels += 1

        results["splits"][split] = {
            "images": img_count,
            "labels": lbl_count,
            "orphaned_images": orphaned_images,
            "orphaned_labels": orphaned_labels,
            "invalid_boxes": invalid_boxes
        }

    return results


def render_dataset_training(core: Dict[str, Any]):
    config = core["config"]
    data_path = Path(config.data_dir)

    st.markdown(
        """
        <div class="main-header" style="border-left-color: #10B981;">
            <h1>📁 Dataset Management & YOLO Training</h1>
            <p>Annotation verification, train/val/test data splits, and model fine-tuning interface.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_data, tab_train = st.tabs(["📊 Dataset Verification", "🏋️ Train Custom Model"])

    with tab_data:
        st.subheader("Automated Dataset Health Audit")
        st.caption(f"Inspecting directory: `{data_path}`")

        if st.button("🔍 Run Dataset Integrity Check", type="primary"):
            audit = validate_yolo_dataset(data_path)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Training Split")
                st.write(audit["splits"].get("train", {}))
            with col2:
                st.markdown("#### Validation Split")
                st.write(audit["splits"].get("val", {}))
            with col3:
                st.markdown("#### Test Split")
                st.write(audit["splits"].get("test", {}))

            st.markdown("#### Annotated Class Distribution")
            st.write(audit["class_counts"])

            # Explain how to structure
            if sum(audit["class_counts"].values()) == 0:
                st.info(
                    "No labeled bounding boxes found yet in `data/labels/`. "
                    "Follow `data/README.md` to populate images and annotations."
                )

    with tab_train:
        st.subheader("YOLO Fine-Tuning Hyperparameters")
        st.warning(
            "⚠️ Training deep neural networks is computationally intensive. "
            "For systems without a dedicated NVIDIA GPU, use small epochs or train in Google Colab."
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            base_model = st.selectbox("Base Architecture", ["yolov8n.pt (Nano - Fast)", "yolov8s.pt (Small)", "yolov8m.pt (Medium)"])
            epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30, step=5)
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        with col_t2:
            img_size = st.selectbox("Inference Resolution (imgsz)", [320, 416, 512, 640], index=3)
            device_train = st.selectbox("Device", ["cpu", "0 (CUDA GPU)"], index=0)
            data_yaml = st.text_input("data.yaml Path", value="data/data.yaml")

        st.markdown(
            f"""
            #### Recommended Command Line Execution:
            To maintain full background stability and monitor GPU telemetry, run the training command directly in your terminal:
            ```bash
            python -m ultralytics train model={base_model.split()[0]} data={data_yaml} epochs={epochs} batch={batch_size} imgsz={img_size} device={device_train.split()[0]} project=models name=fire_smoke_run
            ```
            """
        )
