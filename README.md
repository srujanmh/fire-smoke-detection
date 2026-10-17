# AI-Powered Real-Time Fire and Smoke Detection System Using OpenCV and YOLO

> **Academic Final-Year Capstone Project**  
> *B.Tech Artificial Intelligence and Machine Learning (AI & ML)*

---

## 1. Project Overview

**AegisVision AI** is a production-style, modular computer vision system designed for autonomous, real-time fire and early-stage smoke detection. Built using **OpenCV** and **Ultralytics YOLO**, the system ingests multi-modal visual inputs—including local hardware webcams, uploaded static images, pre-recorded video files, and remote RTSP/HTTP network camera streams—to localize and classify combustion threats.

To mitigate false alarms common in real-world optical monitoring, the system introduces a **temporal confirmation engine** (sliding window buffer) and human-in-the-loop audit logging backed by **SQLite**.

---

## 2. Key Features

- **Multi-Source Ingestion:**
  - **Live Webcam:** Real-time surveillance with OpenCV `cv2.CAP_DSHOW` optimization on Windows.
  - **Static Images:** High-resolution inspection supporting JPG, JPEG, PNG, and WEBP formats with cryptographic magic-byte validation.
  - **Recorded Video:** Memory-efficient, frame-by-frame streaming supporting MP4, AVI, and MOV with configurable sampling intervals.
  - **IP / RTSP Network Streams:** Credential-masked streaming for industrial CCTV and IP cameras.
- **Deep Learning Inference:**
  - Integrated with **Ultralytics YOLO** (YOLOv8/v9/v11).
  - Dynamic class metadata mapping (`model.names` inspection) preventing hardcoded class ID failures.
  - Independent confidence threshold tuning for fire and smoke.
  - Automatic CUDA GPU detection with fallback to CPU inference.
- **Temporal Alert & Defense Architecture:**
  - $M$-of-$N$ sliding temporal window to eliminate transient frame flicker and lighting spikes.
  - Configurable audio alarm cooldown periods.
  - Browser-native Web Audio API synthesizer for zero-dependency cross-platform alert tones.
- **Forensic Audit & Persistence:**
  - Embedded SQLite database with parameterized SQL queries.
  - Automated forensic snapshot watermarking with incident timestamps.
  - False-positive human audit logging (Correct, False Positive, Uncertain).
  - One-click CSV audit report generation.
- **Academic Evaluation Suite:**
  - Ground-truth evaluation using IoU (Intersection over Union) spatial bounding box matching.
  - Calculation of Precision, Recall, F1-Score, and Mean Average Precision (mAP@0.5).
  - Automated YOLO dataset validator checking missing pairs, orphaned files, and degenerate coordinates.

---

## 3. Technology Stack

| Category | Component | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10 / 3.11 | Core runtime environment |
| **Computer Vision** | OpenCV (`opencv-python`) | Hardware frame grabber, video decoding/encoding, box rendering |
| **Image Processing** | Pillow (PIL), NumPy | Matrix arithmetic, image format encoding |
| **Deep Learning** | Ultralytics YOLO, PyTorch | Deep neural network inference, feature extraction |
| **Web UI & Dashboard** | Streamlit | Industrial safety operations center |
| **Visual Analytics** | Plotly Express & Graph Objects | Real-time charts, telemetry distributions |
| **Database** | SQLite3, Pandas | Transactional event history, metrics aggregation |
| **Testing** | Pytest | Automated unit and integration testing |

---

## 4. System Architecture

```
[ Visual Input Stream ]
  │── Live Webcam (cv2.VideoCapture)
  │── Static Image (bytes_to_cv2)
  │── Video File (sequential frame generator)
  └── IP Camera (RTSP / HTTP stream)
        │
        ▼
[ Input Sanitization & Preprocessing ]
  │── Magic-byte header validation
  │── Path traversal protection (sanitize_filename)
  └── Color space conversion (BGR ⇄ RGB)
        │
        ▼
[ Core AI Detector (modules/detector.py) ]
  │── PyTorch / Ultralytics YOLO Engine
  │── Dynamic Class Metadata Extraction
  │── Dual-Threshold Confidence Filter (Fire vs. Smoke)
  └── Non-Maximum Suppression (NMS IoU)
        │
        ▼
[ Temporal Confirmation Filter (modules/alerts.py) ]
  │── Sliding window buffer (e.g. 3 of 5 frames positive)
  └── Audio alarm rate-limiter & Web Audio synthesizer
        │
        ▼
[ Event Manager & Forensic Persistence (modules/database.py) ]
  │── Deduplication rate-limiter
  │── Forensic snapshot stamping (outputs/snapshots/)
  └── Parameterized SQLite insertion (database/detection_events.db)
        │
        ▼
[ Streamlit UI Dashboard (pages/ & app.py) ]
  │── Live video overlay with bounding boxes & badges
  │── Interactive Plotly telemetry charts
  └── False-positive review audit panel
```

---

## 5. Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python:** Python 3.10 or 3.11 (64-bit recommended).
- **Hardware:**
  - *Minimum:* Dual-core CPU, 4 GB RAM, integrated webcam or USB camera.
  - *Recommended:* 4+ core CPU, 8+ GB RAM, NVIDIA GPU with CUDA 11.8+ / 12.1+ for 30+ FPS real-time surveillance.

---

## 6. Installation & Environment Setup

Follow these exact Windows PowerShell instructions to set up the project:

### Step 1: Clone or Navigate to the Project
```powershell
cd C:\Users\sruja\.gemini\antigravity\scratch\fire-smoke-detection
```

### Step 2: Create a Virtual Environment
```powershell
py -3.11 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Model Weights Setup

The application looks for YOLO weights at:
```
models/fire_smoke.pt
```

### How to Obtain or Train Weights:
1. **Option A: Pretrained Checkpoint**
   If you have trained weights (e.g. `best.pt` from Roboflow or Kaggle), copy the file into the `models/` directory and rename it to `fire_smoke.pt`:
   ```powershell
   Copy-Item "C:\path\to\your\best.pt" -Destination "models\fire_smoke.pt"
   ```
2. **Option B: Train with YOLOv8**
   Navigate to the **Dataset & Training** tab in the dashboard, or run:
   ```powershell
   python -m ultralytics train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 project=models name=fire_smoke_run
   ```
3. **Option C: UI Simulation Mode (No Weights Required)**
   If weights have not yet finished training, you can launch the application immediately and toggle **Simulated Mode** in the **Settings** page. This enables full end-to-end testing of the UI, webcam streaming, audio alarms, database writes, and forensic snapshots. Simulated outputs are prominently flagged as `[SIMULATED - NON-AI]`.

---

## 8. Launching the Application

Start the Streamlit operations center:
```powershell
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 9. Operating Instructions by Page

### 📊 Dashboard
- View persistent database totals: Fire Incidents, Smoke Incidents, Images Analyzed, Videos Processed.
- Inspect interactive Plotly charts showing incident ratios, sources, timeline, and confidence distribution.
- Review recent forensic incident snapshots.

### 📹 Live Camera
- Select your hardware camera index (0 is typically the built-in webcam).
- Adjust the detection confidence threshold.
- Toggle audio alarms and adjust temporal window settings.
- Click **Start Surveillance**. The system renders live bounding boxes, displays real-time FPS/latency, and triggers forensic logging when fire/smoke is confirmed.

### 🖼️ Image Detection
- Drag-and-drop any JPG, PNG, or WEBP image.
- View original and annotated images side-by-side.
- Download the annotated image with forensic bounding boxes.
- Conduct a **Quality Assurance Audit**: Mark detections as *Correct*, *False Positive*, or *Uncertain* to build empirical feedback.

### 🎥 Video Detection
- Upload MP4, AVI, or MOV footage.
- Select the **Frame Processing Interval** (e.g., analyze every 2nd frame for 2x faster speed).
- Watch real-time processing progress with frame index and video timestamps.
- Download the fully annotated processed video.

### 🌐 IP Camera Stream
- Enter your RTSP or HTTP camera stream URL (e.g., `rtsp://user:pass@192.168.1.100:554/live`).
- Credentials are automatically scrubbed and never stored in plain-text logs.
- Click **Connect & Start Stream**.

### 📜 Detection History
- Query incidents by class, source, or date range.
- Inspect individual events and view stored forensic snapshots.
- Export all records to CSV via the **Export History to CSV** button.

### 🎯 Model Evaluation
- Run an empirical benchmark against ground-truth YOLO annotations (`data/data.yaml`).
- View exact Precision, Recall, F1-Score, and mAP@0.5 calculated via spatial IoU matching.

### 📁 Dataset & Training
- Validate dataset integrity: detect orphaned images/labels and invalid coordinate boundaries.
- Configure hyperparameters and launch YOLO training.

---

## 10. Running Automated Tests

To run the complete test suite:
```powershell
pytest tests/ -v
```
All unit tests verify:
- Magic byte validation and path traversal sanitization.
- SQLite parameterized insertions, queries, filters, and metrics.
- Temporal sliding window confirmation and rate-limiting deduplication.
- Model missing error handling and class metadata mapping.

---

## 11. Troubleshooting Common Issues

### 1. `ModuleNotFoundError: No module named '...'`
Ensure the virtual environment is active:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Camera Access Denied or Black Screen (`cv2.VideoCapture`)
- In Windows Settings, navigate to **Privacy & Security → Camera** and ensure desktop apps have permission to access the camera.
- Try changing the camera index in the UI from `0` to `1` or `2`.
- Verify no other software (Zoom, Teams, Skype) is currently using the webcam.

### 3. Model Weights Not Found Warning
Place a valid YOLO weights file at `models/fire_smoke.pt`, or open the **Settings** page and enter the custom path to your `.pt` file.

### 4. Video Codec Export Error
If MP4 export fails on your system, the application automatically falls back to the `XVID` codec with an `.avi` container. Ensure `ffmpeg` or `opencv-python` with standard codecs is installed.

### 5. CUDA GPU Not Detected
The system automatically defaults to CPU inference. To use CUDA acceleration, install the CUDA-enabled build of PyTorch:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 12. Privacy, Safety & Ethical Considerations

1. **Assistance Tool Notice:** AegisVision AI is an academic research prototype. It is **not** a certified replacement for building-code-compliant smoke alarms or fire suppression systems.
2. **Local Data Processing:** All visual data, camera streams, and video files are processed strictly on your local machine. No images are sent to external cloud APIs.
3. **Credential Sanitization:** RTSP stream URLs containing embedded authentication credentials (`rtsp://user:pass@host`) are regex-sanitized so passwords never appear in application logs or history tables.

---

## 13. Project Structure

```
fire-smoke-detection/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive documentation & guide
├── .gitignore                  # Git exclusions (weights, DB, outputs)
├── .env.example                # Sample environment variables
│
├── models/
│   └── README.md               # Model format and weights guide
│
├── modules/
│   ├── detector.py             # YOLO detector wrapper with dynamic class mapping
│   ├── camera.py               # Hardware camera management with CAP_DSHOW
│   ├── image_processor.py      # Static image inference & annotation
│   ├── video_processor.py      # Sequential memory-efficient video pipeline
│   ├── ip_camera.py            # RTSP/HTTP network stream client
│   ├── alerts.py               # Temporal confirmation & audio alarms
│   ├── database.py             # SQLite transactional database manager
│   ├── event_manager.py        # Deduplication & forensic snapshot engine
│   └── evaluation.py           # Empirical IoU-based model evaluation
│
├── pages/
│   ├── dashboard.py            # Operations overview & Plotly telemetry
│   ├── live_camera.py          # Real-time webcam sentinel page
│   ├── image_detection.py      # Forensic static image analysis
│   ├── video_detection.py      # Video stream processor & timeline
│   ├── ip_camera_page.py       # Remote CCTV/IP stream monitor
│   ├── history.py              # Incident history, audit review & CSV export
│   ├── analytics.py            # Trend intelligence & confounder guide
│   ├── model_evaluation.py     # Ground-truth evaluation benchmarks
│   ├── dataset_training.py     # Dataset auditor & YOLO training launcher
│   ├── settings.py             # Configuration & hardware tuning
│   └── about.py                # Academic formulations & architecture
│
├── utils/
│   ├── config.py               # Centralized configuration dataclass
│   ├── file_validation.py      # Magic-byte verification & size limits
│   ├── image_utils.py          # Bounding box rendering & snapshot watermarking
│   └── logging_config.py       # Rotating logger with credential scrubbing
│
├── database/                   # SQLite database storage directory
├── data/
│   ├── README.md               # YOLO dataset guide
│   └── data.yaml               # YOLO dataset specification
│
├── outputs/
│   ├── snapshots/              # Forensic incident images
│   ├── processed_videos/       # Exported annotated videos
│   └── reports/                # Exported CSV history reports
│
└── tests/
    ├── test_detector.py        # Detector error handling & class mapping tests
    ├── test_database.py        # SQLite queries, filters, and metrics tests
    ├── test_file_validation.py  # File validation & URL sanitization tests
    └── test_event_manager.py   # Temporal window & rate-limiting tests
```
