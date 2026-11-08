"""
AI-Powered Real-Time Fire & Smoke Detection System
Main Application Entry Point (Streamlit).
Industrial safety monitoring dashboard with modular multi-page navigation.
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Auto-download model on Streamlit Cloud if MODEL_URL env var is set
try:
    from download_model import ensure_model
    ensure_model()
except Exception:
    pass

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_default_config, AppConfig
from utils.logging_config import logger
from modules.detector import FireSmokeDetector
from modules.database import DatabaseManager
from modules.alerts import AlertSystem
from modules.event_manager import EventManager
from modules.camera import CameraManager
from modules.ip_camera import IPCameraManager
from modules.image_processor import ImageProcessor
from modules.video_processor import VideoProcessor
from modules.evaluation import ModelEvaluator

# Initialize Streamlit Page Configuration
st.set_page_config(
    page_title="AI Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Safety CSS Theme
CUSTOM_CSS = """
<style>
    /* Global Theme Styles */
    .stApp {
        background-color: #111827;
        color: #F9FAFB;
    }
    
    /* Top Header Bar */
    .main-header {
        background: linear-gradient(90deg, #1F2937 0%, #111827 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #EF4444;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #F9FAFB;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .main-header p {
        color: #9CA3AF;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status Badges */
    .badge-normal {
        background-color: #064E3B;
        color: #34D399;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #059669;
        display: inline-block;
    }
    .badge-fire {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid #DC2626;
        animation: pulse 1.5s infinite;
        display: inline-block;
    }
    .badge-smoke {
        background-color: #78350F;
        color: #FBBF24;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid #D97706;
        display: inline-block;
    }
    .badge-simulated {
        background-color: #312E81;
        color: #A5B4FC;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px dashed #6366F1;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1F2937;
        border-right: 1px solid #374151;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def init_application_core():
    """Initialize persistent backend services once across app lifetime."""
    config = get_default_config()
    db_manager = DatabaseManager(config.database_path)
    alert_system = AlertSystem(
        cooldown_seconds=config.alert_cooldown_seconds,
        temporal_window_size=config.temporal_window_frames,
        temporal_min_detections=config.temporal_min_detections,
        audio_enabled=config.enable_audio_alerts
    )
    event_manager = EventManager(
        db_manager=db_manager,
        alert_system=alert_system,
        snapshots_dir=config.snapshots_dir,
        log_cooldown_seconds=config.alert_cooldown_seconds
    )
    detector = FireSmokeDetector(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        fire_threshold=config.fire_confidence_threshold,
        smoke_threshold=config.smoke_confidence_threshold,
        iou_threshold=config.iou_threshold,
        imgsz=config.inference_img_size,
        device="cpu" if config.force_cpu else None
    )
    image_processor = ImageProcessor(detector, event_manager)
    video_processor = VideoProcessor(detector, event_manager, config.processed_videos_dir)
    evaluator = ModelEvaluator(detector)

    return {
        "config": config,
        "db": db_manager,
        "alerts": alert_system,
        "events": event_manager,
        "detector": detector,
        "image_proc": image_processor,
        "video_proc": video_processor,
        "evaluator": evaluator
    }


# Load Core Services
core = init_application_core()
config: AppConfig = core["config"]
db: DatabaseManager = core["db"]
alerts: AlertSystem = core["alerts"]
events: EventManager = core["events"]
detector: FireSmokeDetector = core["detector"]
image_proc: ImageProcessor = core["image_proc"]
video_proc: VideoProcessor = core["video_proc"]
evaluator: ModelEvaluator = core["evaluator"]

# Session-specific camera instances
if "camera_manager" not in st.session_state:
    st.session_state.camera_manager = CameraManager(config.default_camera_index)
if "ip_camera_manager" not in st.session_state:
    st.session_state.ip_camera_manager = IPCameraManager()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(
    """
    <div style="padding: 10px 0; border-bottom: 1px solid #374151; margin-bottom: 15px;">
        <h2 style="margin: 0; color: #EF4444; font-size: 1.4rem;">🔥 AegisVision AI</h2>
        <span style="color: #9CA3AF; font-size: 0.8rem;">Real-Time Fire & Smoke Sentinel</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Model Status Pill in Sidebar
hw = detector.get_hardware_info()
if detector.is_loaded:
    st.sidebar.markdown(
        f"""
        <div style="background-color: #064E3B; padding: 8px 12px; border-radius: 6px; border: 1px solid #059669; margin-bottom: 15px;">
            <div style="color: #34D399; font-weight: 600; font-size: 0.85rem;">● Model: Ready</div>
            <div style="color: #9CA3AF; font-size: 0.75rem;">Device: {hw['active_device'].upper()} ({hw['device_name']})</div>
        </div>
        """,
        unsafe_allow_html=True
    )
elif detector.is_simulation_mode:
    st.sidebar.markdown(
        """
        <div style="background-color: #312E81; padding: 8px 12px; border-radius: 6px; border: 1px dashed #6366F1; margin-bottom: 15px;">
            <div style="color: #A5B4FC; font-weight: 600; font-size: 0.85rem;">⚙ Mode: SIMULATED (Non-AI)</div>
            <div style="color: #9CA3AF; font-size: 0.75rem;">Demonstration / UI Test</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style="background-color: #451A03; padding: 8px 12px; border-radius: 6px; border: 1px solid #B45309; margin-bottom: 15px;">
            <div style="color: #FBBF24; font-weight: 600; font-size: 0.85rem;">⚠ Model: Not Loaded</div>
            <div style="color: #9CA3AF; font-size: 0.75rem;">Place weights in models/ or use Settings</div>
        </div>
        """,
        unsafe_allow_html=True
    )

PAGES = [
    "📊 Dashboard",
    "📹 Live Camera",
    "🖼️ Image Detection",
    "🎥 Video Detection",
    "🌐 IP Camera Stream",
    "📜 Detection History",
    "📈 Analytics & Trends",
    "🎯 Model Evaluation",
    "📁 Dataset & Training",
    "⚙️ Settings",
    "ℹ️ About System"
]

selected_page = st.sidebar.radio("Navigation", PAGES)

# Safety System Disclaimer in sidebar footer
st.sidebar.markdown(
    """
    <div style="position: relative; margin-top: 40px; padding: 10px; background-color: #111827; border-radius: 6px; border: 1px solid #374151; font-size: 0.75rem; color: #9CA3AF;">
        <strong style="color: #F59E0B;">Notice:</strong> This software is an AI assistance prototype for research and education. It is <em>not</em> a certified life-safety fire alarm.
    </div>
    """,
    unsafe_allow_html=True
)

# Route to page views
if selected_page == "📊 Dashboard":
    from pages.dashboard import render_dashboard
    render_dashboard(core)
elif selected_page == "📹 Live Camera":
    from pages.live_camera import render_live_camera
    render_live_camera(core, st.session_state.camera_manager)
elif selected_page == "🖼️ Image Detection":
    from pages.image_detection import render_image_detection
    render_image_detection(core)
elif selected_page == "🎥 Video Detection":
    from pages.video_detection import render_video_detection
    render_video_detection(core)
elif selected_page == "🌐 IP Camera Stream":
    from pages.ip_camera_page import render_ip_camera
    render_ip_camera(core, st.session_state.ip_camera_manager)
elif selected_page == "📜 Detection History":
    from pages.history import render_history
    render_history(core)
elif selected_page == "📈 Analytics & Trends":
    from pages.analytics import render_analytics
    render_analytics(core)
elif selected_page == "🎯 Model Evaluation":
    from pages.model_evaluation import render_model_evaluation
    render_model_evaluation(core)
elif selected_page == "📁 Dataset & Training":
    from pages.dataset_training import render_dataset_training
    render_dataset_training(core)
elif selected_page == "⚙️ Settings":
    from pages.settings import render_settings
    render_settings(core)
elif selected_page == "ℹ️ About System":
    from pages.about import render_about
    render_about(core)
