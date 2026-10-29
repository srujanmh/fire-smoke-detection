"""
FastAPI Modern Web Dashboard for AI Fire & Smoke Detection.
Alternative web interface with ZERO Streamlit.
Provides MJPEG live camera streaming, REST endpoints, and an industrial dark dashboard.
"""

import os
import sys
import time
import cv2
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
import uvicorn

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_default_config
from utils.image_utils import draw_detections, bgr_to_rgb, rgb_to_bgr, cv2_to_bytes
from modules.detector import FireSmokeDetector
from modules.database import DatabaseManager
from modules.alerts import AlertSystem
from modules.event_manager import EventManager
from modules.camera import CameraManager
from modules.image_processor import ImageProcessor

app = FastAPI(title="AegisVision AI Sentinel API")

# Initialize Core Services
config = get_default_config()
db = DatabaseManager(config.database_path)
alerts = AlertSystem(
    cooldown_seconds=config.alert_cooldown_seconds,
    temporal_window_size=config.temporal_window_frames,
    temporal_min_detections=config.temporal_min_detections,
    audio_enabled=config.enable_audio_alerts
)
event_manager = EventManager(
    db_manager=db,
    alert_system=alerts,
    snapshots_dir=config.snapshots_dir,
    log_cooldown_seconds=config.alert_cooldown_seconds
)
detector = FireSmokeDetector(
    model_path=config.model_path,
    confidence_threshold=0.45,
    fire_threshold=0.45,
    smoke_threshold=0.45,
    iou_threshold=config.iou_threshold,
    imgsz=config.inference_img_size
)
detector.is_simulation_mode = False

camera = CameraManager(config.default_camera_index)
image_proc = ImageProcessor(detector, event_manager)

# Embedded Single-Page Industrial Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AegisVision AI — Fire & Smoke Operations Center</title>
    <style>
        :root {
            --bg-dark: #111827;
            --card-bg: #1F2937;
            --border: #374151;
            --fire: #EF4444;
            --smoke: #F59E0B;
            --safe: #22C55E;
            --text-primary: #F9FAFB;
            --text-secondary: #9CA3AF;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-dark); color: var(--text-primary); display: flex; height: 100vh; overflow: hidden; }
        
        /* Sidebar */
        aside { width: 250px; background: var(--card-bg); border-right: 1px solid var(--border); padding: 24px 16px; display: flex; flex-direction: column; gap: 10px; }
        .logo { font-size: 1.3rem; font-weight: 800; color: var(--fire); display: flex; align-items: center; gap: 8px; margin-bottom: 24px; }
        .nav-btn { background: transparent; border: none; color: var(--text-secondary); text-align: left; padding: 12px 14px; border-radius: 6px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .nav-btn:hover, .nav-btn.active { background: #374151; color: var(--text-primary); }
        
        /* Main Workspace */
        main { flex: 1; padding: 24px 32px; overflow-y: auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .header h1 { font-size: 1.6rem; font-weight: 700; }
        
        /* Panels */
        .panel { display: none; }
        .panel.active { display: block; }
        
        /* Video Stream Container */
        .video-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 480px; position: relative; }
        .video-box img { max-width: 100%; height: auto; display: block; }
        
        /* Metric Cards */
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .metric-card { background: var(--card-bg); border: 1px solid var(--border); padding: 18px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
        .metric-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        
        /* Buttons */
        .btn { background: #3B82F6; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn-danger { background: var(--fire); }
        .btn-success { background: var(--safe); }
        
        /* Table */
        table { width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; margin-top: 16px; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
        th { background: #2D3748; color: var(--text-secondary); font-weight: 600; }
        
        .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .badge-fire { background: #7F1D1D; color: #F87171; border: 1px solid #DC2626; }
        .badge-smoke { background: #78350F; color: #FBBF24; border: 1px solid #D97706; }
        .badge-normal { background: #064E3B; color: #34D399; border: 1px solid #059669; }
    </style>
</head>
<body>
    <aside>
        <div class="logo">🔥 AegisVision AI</div>
        <button class="nav-btn active" onclick="switchTab('live')">📹 Live Camera</button>
        <button class="nav-btn" onclick="switchTab('image')">🖼️ Image Analysis</button>
        <button class="nav-btn" onclick="switchTab('history')">📜 Incident History</button>
        <button class="nav-btn" onclick="switchTab('settings')">⚙️ System Settings</button>
        <div style="margin-top: auto; font-size: 0.75rem; color: var(--text-secondary); padding: 10px; background: #111827; border-radius: 6px;">
            Standalone FastAPI Engine<br>Zero Streamlit Overhead
        </div>
    </aside>

    <main>
        <!-- LIVE CAMERA PANEL -->
        <div id="panel-live" class="panel active">
            <div class="header">
                <h1>Real-Time Webcam Sentinel</h1>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-danger" onclick="toggleStream(true)">▶ Start Camera</button>
                    <button class="btn" style="background:#4B5563;" onclick="toggleStream(false)">⏹ Stop Camera</button>
                </div>
            </div>
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-value" id="val-status" style="color: var(--safe); font-size: 1.2rem;">AI ACTIVE (YOLOv8)</div><div class="metric-label">Model Engine</div></div>
                <div class="metric-card"><div class="metric-value" id="val-fps" style="color: #60A5FA;">0.0</div><div class="metric-label">Stream FPS</div></div>
                <div class="metric-card"><div class="metric-value" id="val-latency" style="color: #A78BFA;">0.0 ms</div><div class="metric-label">Latency</div></div>
                <div class="metric-card"><div class="metric-value" id="val-events" style="color: var(--fire);">0</div><div class="metric-label">Events Logged</div></div>
            </div>
            <div class="video-box" id="stream-box">
                <p id="placeholder-text" style="color: var(--text-secondary);">Camera stream offline. Click "Start Camera" above.</p>
                <img id="camera-stream" src="" style="display: none;" />
            </div>
        </div>

        <!-- IMAGE ANALYSIS PANEL -->
        <div id="panel-image" class="panel">
            <div class="header">
                <h1>Static Image Detection</h1>
            </div>
            <div style="background: var(--card-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 20px;">
                <input type="file" id="img-input" accept="image/*" style="display: none;" onchange="uploadImage()" />
                <button class="btn" onclick="document.getElementById('img-input').click()">📂 Select Image (.jpg, .png, .webp)</button>
                <span id="img-filename" style="margin-left: 15px; color: var(--text-secondary);">No file chosen</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="video-box" style="min-height: 380px;">
                    <img id="orig-img" style="display: none;" />
                    <p id="orig-txt" style="color: var(--text-secondary);">Original Image</p>
                </div>
                <div class="video-box" style="min-height: 380px;">
                    <img id="annot-img" style="display: none;" />
                    <p id="annot-txt" style="color: var(--text-secondary);">Annotated Threat Detections</p>
                </div>
            </div>
            <div id="img-summary-box" style="margin-top: 15px; background: var(--card-bg); padding: 15px; border-radius: 8px; display: none;"></div>
        </div>

        <!-- HISTORY PANEL -->
        <div id="panel-history" class="panel">
            <div class="header">
                <h1>Incident Audit History</h1>
                <div>
                    <button class="btn" onclick="loadHistory()">🔄 Refresh</button>
                    <a href="/api/export_csv" class="btn btn-success" style="text-decoration: none; display: inline-block;">📥 Export CSV</a>
                </div>
            </div>
            <div id="history-table-container">Loading incident logs...</div>
        </div>

        <!-- SETTINGS PANEL -->
        <div id="panel-settings" class="panel">
            <div class="header"><h1>System Settings</h1></div>
            <div style="background: var(--card-bg); padding: 24px; border-radius: 8px; border: 1px solid var(--border); max-width: 600px;">
                <div style="margin-bottom: 20px;">
                    <label style="display: block; font-weight: 600; margin-bottom: 8px;">Confidence Threshold</label>
                    <input type="range" id="conf-slider" min="0.1" max="0.95" step="0.05" value="0.35" oninput="updateConfidence(this.value)" style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 0.85rem;">
                        <span>0.10 (High Sensitivity)</span><span id="conf-val" style="color: var(--text-primary); font-weight: 700;">0.35</span><span>0.95 (High Specificity)</span>
                    </div>
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display: flex; align-items: center; gap: 10px; cursor: pointer;">
                        <input type="checkbox" id="sim-check" onchange="toggleSimMode(this.checked)" style="width: 18px; height: 18px;">
                        <span>Enable Simulated Mode (Test UI/Audio without AI weights)</span>
                    </label>
                </div>
            </div>
        </div>
    </main>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('panel-' + tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'history') loadHistory();
        }

        function toggleStream(start) {
            const streamImg = document.getElementById('camera-stream');
            const placeholder = document.getElementById('placeholder-text');
            if (start) {
                streamImg.src = "/video_feed?t=" + new Date().getTime();
                streamImg.style.display = "block";
                placeholder.style.display = "none";
                document.getElementById('val-status').innerText = "STREAMING";
            } else {
                streamImg.src = "";
                streamImg.style.display = "none";
                placeholder.style.display = "block";
                document.getElementById('val-status').innerText = "OFFLINE";
            }
        }

        async function uploadImage() {
            const input = document.getElementById('img-input');
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            document.getElementById('img-filename').innerText = file.name;

            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/detect_image", { method: "POST", body: formData });
            const data = await res.json();

            if (data.success) {
                const origImg = document.getElementById('orig-img');
                const annotImg = document.getElementById('annot-img');
                origImg.src = data.original_b64;
                origImg.style.display = "block";
                document.getElementById('orig-txt').style.display = "none";

                annotImg.src = data.annotated_b64;
                annotImg.style.display = "block";
                document.getElementById('annot-txt').style.display = "none";

                const summaryBox = document.getElementById('img-summary-box');
                summaryBox.style.display = "block";
                summaryBox.innerHTML = `<strong>Status:</strong> ${data.status} | <strong>Fire Objects:</strong> ${data.summary.fire_count} | <strong>Smoke Objects:</strong> ${data.summary.smoke_count} | <strong>Inference Time:</strong> ${data.summary.processing_time_ms.toFixed(1)} ms`;
            }
        }

        async function loadHistory() {
            const res = await fetch("/api/events");
            const events = await res.json();
            const container = document.getElementById('history-table-container');

            if (!events || events.length === 0) {
                container.innerHTML = "<p style='color: var(--text-secondary); padding: 20px;'>No detection incidents logged yet.</p>";
                return;
            }

            let html = "<table><thead><tr><th>ID</th><th>Timestamp</th><th>Source</th><th>Class</th><th>Confidence</th><th>Status</th></tr></thead><tbody>";
            events.forEach(ev => {
                const badgeClass = ev.detected_class.includes('fire') ? 'badge-fire' : 'badge-smoke';
                html += `<tr>
                    <td>#${ev.event_id}</td>
                    <td>${ev.timestamp}</td>
                    <td>${ev.source}</td>
                    <td><span class="badge ${badgeClass}">${ev.detected_class}</span></td>
                    <td>${(ev.confidence * 100).toFixed(1)}%</td>
                    <td>${ev.alert_status}</td>
                </tr>`;
            });
            html += "</tbody></table>";
            container.innerHTML = html;
        }

        async function toggleSimMode(enabled) {
            await fetch("/api/settings?simulated=" + enabled, { method: "POST" });
        }

        async function updateConfidence(val) {
            document.getElementById('conf-val').innerText = val;
            await fetch("/api/settings?conf=" + val, { method: "POST" });
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page industrial web dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


def generate_camera_mjpeg():
    """Generator for live MJPEG video stream."""
    camera.start()
    try:
        while True:
            ret, frame_rgb = camera.read_frame()
            if not ret or frame_rgb is None:
                time.sleep(0.04)
                continue

            det_result = detector.detect(frame_rgb, is_rgb=True)
            event_manager.process_incident(frame_rgb, det_result, source="webcam")
            annotated_rgb = draw_detections(frame_rgb, det_result.get("detections", []), is_rgb=True)
            annotated_bgr = rgb_to_bgr(annotated_rgb)

            success, jpeg_bytes = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes.tobytes() + b"\r\n"
            )
            time.sleep(0.03)  # ~30 FPS
    finally:
        camera.stop()


@app.get("/video_feed")
def video_feed():
    """MJPEG streaming endpoint for HTML5 <img> tags."""
    return StreamingResponse(
        generate_camera_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/detect_image")
async def detect_image_api(file: UploadFile = File(...)):
    """Run YOLO threat detection on uploaded static image."""
    import base64
    content = await file.read()
    result = image_proc.process_image(content, file.filename, save_incident=True)

    if not result["success"]:
        return JSONResponse({"success": False, "error": result.get("error")}, status_code=400)

    orig_b64 = f"data:image/jpeg;base64,{base64.b64encode(cv2_to_bytes(rgb_to_bgr(result['original_image']))).decode('utf-8')}"
    annot_b64 = f"data:image/jpeg;base64,{base64.b64encode(result['download_bytes']).decode('utf-8')}"

    return {
        "success": True,
        "status": result["status"],
        "summary": result["summary"],
        "original_b64": orig_b64,
        "annotated_b64": annot_b64
    }


@app.get("/api/events")
def get_events_api():
    """Retrieve the 100 most recent detection events from SQLite."""
    return db.get_events(limit=100)


@app.get("/api/export_csv")
def export_csv_api():
    """Export database events to CSV and trigger file download."""
    csv_path = os.path.join(config.reports_dir, "fastapi_export.csv")
    db.export_to_csv(csv_path)
    return FileResponse(csv_path, media_type="text/csv", filename="fire_smoke_incidents.csv")


@app.post("/api/settings")
def update_settings(simulated: Optional[bool] = None, conf: Optional[float] = None):
    """Update runtime settings."""
    if simulated is not None:
        detector.is_simulation_mode = simulated
    if conf is not None:
        detector.confidence_threshold = conf
        detector.fire_threshold = conf
        detector.smoke_threshold = max(conf, 0.40)
    return {"success": True, "simulated": detector.is_simulation_mode, "confidence": detector.confidence_threshold}


if __name__ == "__main__":
    print("\nStarting AegisVision AI FastAPI Sentinel at http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
