"""
About the Project & Academic Architecture Page.
Details the research methodology, mathematical formulations,
system architecture, and academic citations.
"""

import streamlit as st
from typing import Dict, Any


def render_about(core: Dict[str, Any]):
    st.markdown(
        """
        <div class="main-header" style="border-left-color: #3B82F6;">
            <h1>ℹ️ About AegisVision AI Project</h1>
            <p>B.Tech Artificial Intelligence & Machine Learning Final-Year Capstone Project.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ## 1. Project Abstract
        Rapid, automated detection of early-stage fire and smoke is critical for minimizing casualty risks, industrial downtime,
        and catastrophic environmental damage. Conventional ionization and photoelectric smoke alarms suffer from physical
        diffusion delays and cannot monitor large outdoor spaces, warehouses, or high-ceiling facilities.
        
        This academic project presents **AegisVision AI**, a modular, real-time computer vision system combining **OpenCV**
        for hardware abstraction and forensic image manipulation with **Ultralytics YOLO** for deep convolutional object detection.
        To overcome false-positive susceptibility in uncontrolled optical environments, the system incorporates a **temporal
        confirmation filter** and human-in-the-loop audit logging backed by **SQLite**.

        ---

        ## 2. Theoretical & Mathematical Foundations

        ### A. Object Detection Formulation
        Each predicted bounding box is parameterized as a vector:
        $$\mathbf{b} = [x_{\text{center}}, y_{\text{center}}, w, h, c, p_1, p_2, \dots, p_C]$$
        where:
        - $(x_{\text{center}}, y_{\text{center}})$ denotes the normalized centroid coordinates.
        - $(w, h)$ represents the normalized box width and height relative to image dimensions.
        - $c \in [0, 1]$ represents the objectness confidence score.
        - $p_k$ is the conditional class probability for category $k \in \{\text{Fire}, \text{Smoke}\}$.

        ### B. Intersection over Union (IoU) Metric
        Given a predicted bounding box $B_{\text{pred}}$ and a ground-truth annotation $B_{\text{gt}}$:
        $$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
        A detection is evaluated as a True Positive (TP) if $\text{IoU} \ge 0.50$ and the predicted class label matches ground truth.

        ### C. Precision, Recall, and mAP
        $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
        $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
        $$\text{mAP} = \frac{1}{|C|} \sum_{c \in C} \int_0^1 P_c(R) \, dR$$

        ---

        ## 3. High-Level Architecture Pipeline

        ```
        [ Input Sources ]
          ├── Live Webcam (cv2.CAP_DSHOW)
          ├── Uploaded Static Image (JPG, PNG, WEBP)
          ├── Uploaded Video (MP4, AVI, MOV)
          └── Network Stream (RTSP / HTTP)
                   │
                   ▼
        [ Validation & Preprocessing Layer ]
          ├── Magic-byte verification & size limit checks
          ├── BGR to RGB color conversion
          └── Letterbox scaling (640x640)
                   │
                   ▼
        [ YOLO Neural Network Engine ]
          ├── PyTorch Backbone & CSP-Darknet Feature Pyramid
          ├── Dynamic Class Mapping (Metadata inspection)
          └── Non-Maximum Suppression (IoU >= 0.45)
                   │
                   ▼
        [ Temporal Confirmation & Event Layer ]
          ├── Sliding window buffer (M of N frames confirmed)
          ├── Audio alert cooldown rate limiter
          ├── SQLite transactional event persistence
          └── Forensic snapshot image generation
                   │
                   ▼
        [ Streamlit Industrial Dashboard ]
          ├── Telemetry & latency monitor
          ├── False-positive review audit workflow
          └── Interactive Plotly analytics
        ```

        ---

        ## 4. Academic Future Extensions
        1. **Multispectral Fusion:** Combining visible optical streams with Long-Wave Infrared (LWIR) thermal cameras.
        2. **Edge Hardware Acceleration:** Compiling YOLO weights to ONNX Runtime / TensorRT for deployment on NVIDIA Jetson or Raspberry Pi 5 with Coral Edge TPU.
        3. **Autonomous Aerial Patrol:** Integrating with PX4 / ROS drones for wildfire surveillance.
        """
    )
