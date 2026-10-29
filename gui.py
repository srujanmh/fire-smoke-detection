"""
Native Desktop GUI for AI Fire & Smoke Detection System.
Built with CustomTkinter & OpenCV.
Zero Streamlit dependency, zero browser overhead, direct GPU/CPU camera rendering.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_default_config, AppConfig
from utils.logging_config import logger
from utils.image_utils import draw_detections, bytes_to_cv2, bgr_to_rgb, rgb_to_bgr, cv2_to_bytes
from modules.detector import FireSmokeDetector
from modules.database import DatabaseManager
from modules.alerts import AlertSystem
from modules.event_manager import EventManager
from modules.camera import CameraManager
from modules.image_processor import ImageProcessor
from modules.video_processor import VideoProcessor

# Set CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class FireSmokeApp(ctk.CTk):
    """Modern Desktop GUI Application for Fire & Smoke Detection."""

    def __init__(self):
        super().__init__()

        self.title("AegisVision AI — Real-Time Fire & Smoke Sentinel")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # Initialize Backend Core
        self.config = get_default_config()
        self.db = DatabaseManager(self.config.database_path)
        self.alerts = AlertSystem(
            cooldown_seconds=self.config.alert_cooldown_seconds,
            temporal_window_size=self.config.temporal_window_frames,
            temporal_min_detections=self.config.temporal_min_detections,
            audio_enabled=self.config.enable_audio_alerts
        )
        self.event_manager = EventManager(
            db_manager=self.db,
            alert_system=self.alerts,
            snapshots_dir=self.config.snapshots_dir,
            log_cooldown_seconds=self.config.alert_cooldown_seconds
        )
        self.detector = FireSmokeDetector(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            fire_threshold=self.config.fire_confidence_threshold,
            smoke_threshold=self.config.smoke_confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            imgsz=self.config.inference_img_size,
            device="cpu" if self.config.force_cpu else None
        )
        self.camera = CameraManager(self.config.default_camera_index)
        self.image_proc = ImageProcessor(self.detector, self.event_manager)
        self.video_proc = VideoProcessor(self.detector, self.event_manager, self.config.processed_videos_dir)

        # Video Thread Control
        self.is_camera_running = False
        self.camera_thread: Optional[threading.Thread] = None

        # Build Interface
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self):
        # Grid layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#1F2937")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🔥 AegisVision AI",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#EF4444"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.sub_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Fire & Smoke Sentinel",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF"
        )
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Navigation Buttons
        self.nav_btns = {}
        pages = [
            ("📹 Live Camera", self.show_live_camera),
            ("🖼️ Image Detection", self.show_image_detection),
            ("🎥 Video Detection", self.show_video_detection),
            ("📜 Incident History", self.show_history),
            ("⚙️ Settings", self.show_settings)
        ]

        for i, (name, cmd) in enumerate(pages):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=name,
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="transparent",
                text_color="#F9FAFB",
                hover_color="#374151",
                anchor="w",
                height=40,
                command=cmd
            )
            btn.grid(row=2 + i, column=0, padx=15, pady=5, sticky="ew")
            self.nav_btns[name] = btn

        # Status Card at Bottom of Sidebar
        hw = self.detector.get_hardware_info()
        self.hw_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=f"Device: {hw['active_device'].upper()}\nModel: {'Loaded' if self.detector.is_loaded else 'Missing/Sim'}",
            font=ctk.CTkFont(size=11),
            text_color="#10B981" if self.detector.is_loaded else "#F59E0B",
            justify="left"
        )
        self.hw_label.grid(row=9, column=0, padx=20, pady=20, sticky="s")

        # --- Main Content Area ---
        self.main_container = ctk.CTkFrame(self, fg_color="#111827", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Page Frames
        self.pages = {}
        self.pages["live"] = self._create_live_page()
        self.pages["image"] = self._create_image_page()
        self.pages["video"] = self._create_video_page()
        self.pages["history"] = self._create_history_page()
        self.pages["settings"] = self._create_settings_page()

        # Start on Live Camera Page
        self.show_live_camera()

    def _select_nav(self, active_name: str):
        for name, btn in self.nav_btns.items():
            if name == active_name:
                btn.configure(fg_color="#374151")
            else:
                btn.configure(fg_color="transparent")

    def _hide_all_pages(self):
        for p in self.pages.values():
            p.grid_forget()

    # ==========================================
    # PAGE 1: LIVE CAMERA
    # ==========================================
    def _create_live_page(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container, fg_color="#111827")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=3)
        frame.grid_columnconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            frame,
            text="Real-Time Webcam Sentinel",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#F9FAFB"
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Video Display Canvas / Label
        self.video_display = ctk.CTkLabel(
            frame,
            text="Camera Offline\nClick 'Start Surveillance' to begin",
            fg_color="#1F2937",
            corner_radius=8,
            font=ctk.CTkFont(size=16)
        )
        self.video_display.grid(row=1, column=0, sticky="nsew", padx=(0, 15), pady=5)

        # Side Telemetry Panel
        tele_panel = ctk.CTkFrame(frame, fg_color="#1F2937", corner_radius=8)
        tele_panel.grid(row=1, column=1, sticky="nsew", pady=5)
        tele_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tele_panel, text="Threat Telemetry", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15, padx=15, anchor="w")

        # Threat Status Badge
        self.live_status_badge = ctk.CTkLabel(
            tele_panel,
            text="NORMAL",
            fg_color="#064E3B",
            text_color="#34D399",
            corner_radius=6,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.live_status_badge.pack(fill="x", padx=15, pady=5)

        # Telemetry Text
        self.tele_fps = ctk.CTkLabel(tele_panel, text="FPS: 0.0", anchor="w", text_color="#9CA3AF")
        self.tele_fps.pack(fill="x", padx=15, pady=3)

        self.tele_latency = ctk.CTkLabel(tele_panel, text="Inference: 0.0 ms", anchor="w", text_color="#9CA3AF")
        self.tele_latency.pack(fill="x", padx=15, pady=3)

        self.tele_buffer = ctk.CTkLabel(tele_panel, text="Fire Buffer: 0/5", anchor="w", text_color="#9CA3AF")
        self.tele_buffer.pack(fill="x", padx=15, pady=3)

        self.tele_sim = ctk.CTkLabel(tele_panel, text="", anchor="w", text_color="#A5B4FC")
        self.tele_sim.pack(fill="x", padx=15, pady=3)

        # Controls
        ctk.CTkLabel(tele_panel, text="Controls", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5), padx=15, anchor="w")

        self.start_cam_btn = ctk.CTkButton(
            tele_panel,
            text="▶ Start Surveillance",
            fg_color="#EF4444",
            hover_color="#DC2626",
            command=self.toggle_camera
        )
        self.start_cam_btn.pack(fill="x", padx=15, pady=8)

        self.reset_buf_btn = ctk.CTkButton(
            tele_panel,
            text="🔄 Reset Alert Buffer",
            fg_color="#374151",
            hover_color="#4B5563",
            command=self.reset_alert_buffer
        )
        self.reset_buf_btn.pack(fill="x", padx=15, pady=5)

        return frame

    def toggle_camera(self):
        if not self.is_camera_running:
            success, msg = self.camera.start()
            if not success:
                messagebox.showerror("Camera Error", msg)
                return

            self.is_camera_running = True
            self.start_cam_btn.configure(text="⏹ Stop Camera", fg_color="#4B5563", hover_color="#374151")
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
        else:
            self.is_camera_running = False
            self.camera.stop()
            self.start_cam_btn.configure(text="▶ Start Surveillance", fg_color="#EF4444", hover_color="#DC2626")
            self.video_display.configure(image="", text="Camera Standby")
            self.live_status_badge.configure(text="OFFLINE", fg_color="#374151", text_color="#9CA3AF")

    def _camera_loop(self):
        while self.is_camera_running:
            ret, frame_rgb = self.camera.read_frame()
            if not ret or frame_rgb is None:
                time.sleep(0.05)
                continue

            # Run Detection
            det_result = self.detector.detect(frame_rgb, is_rgb=True)
            incident = self.event_manager.process_incident(frame_rgb, det_result, source="webcam")
            alert_state = incident["alert_state"]
            alert_info = incident["alert_info"]

            # Annotate
            annotated = draw_detections(frame_rgb, det_result.get("detections", []), is_rgb=True)

            # Resize to fit display label
            h, w = annotated.shape[:2]
            target_w = 720
            target_h = int(h * (target_w / w))
            resized = cv2.resize(annotated, (target_w, target_h))

            # Convert to CTkImage
            pil_img = Image.fromarray(resized)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

            # Update GUI safely in main loop
            self.after(0, self._update_camera_ui, ctk_img, det_result, alert_state, alert_info)
            time.sleep(0.01)

    def _update_camera_ui(self, ctk_img, det_result, alert_state, alert_info):
        if not self.is_camera_running:
            return
        self.video_display.configure(image=ctk_img, text="")

        # Badge colors
        if "FIRE" in alert_state:
            self.live_status_badge.configure(text=alert_state, fg_color="#7F1D1D", text_color="#F87171")
        elif "SMOKE" in alert_state:
            self.live_status_badge.configure(text=alert_state, fg_color="#78350F", text_color="#FBBF24")
        elif alert_state == "UNCERTAIN":
            self.live_status_badge.configure(text=alert_state, fg_color="#78350F", text_color="#FBBF24")
        else:
            self.live_status_badge.configure(text="NORMAL", fg_color="#064E3B", text_color="#34D399")

        stats = self.camera.get_stats()
        self.tele_fps.configure(text=f"FPS: {stats['fps']}")
        self.tele_latency.configure(text=f"Inference: {det_result.get('inference_time_ms', 0):.1f} ms")
        self.tele_buffer.configure(text=f"Fire: {alert_info['fire_buffer_count']}/{alert_info['buffer_capacity']} | Smoke: {alert_info['smoke_buffer_count']}/{alert_info['buffer_capacity']}")
        self.tele_sim.configure(text="[SIMULATED - NON-AI]" if det_result.get("is_simulated") else "")

    def reset_alert_buffer(self):
        self.alerts.reset_state()
        self.live_status_badge.configure(text="NORMAL", fg_color="#064E3B", text_color="#34D399")

    # ==========================================
    # PAGE 2: IMAGE DETECTION
    # ==========================================
    def _create_image_page(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container, fg_color="#111827")
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(frame, text="Static Image Threat Analysis", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F9FAFB")
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.upload_img_btn = ctk.CTkButton(
            btn_row,
            text="📂 Choose Image (.jpg, .png, .webp)",
            fg_color="#3B82F6",
            hover_color="#2563EB",
            command=self.upload_and_analyze_image
        )
        self.upload_img_btn.pack(side="left", padx=(0, 10))

        self.img_status_lbl = ctk.CTkLabel(btn_row, text="No image selected", text_color="#9CA3AF")
        self.img_status_lbl.pack(side="left")

        # Side by side previews
        self.orig_img_display = ctk.CTkLabel(frame, text="Original Image", fg_color="#1F2937", corner_radius=8)
        self.orig_img_display.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=5)

        self.annotated_img_display = ctk.CTkLabel(frame, text="Annotated Detections", fg_color="#1F2937", corner_radius=8)
        self.annotated_img_display.grid(row=2, column=1, sticky="nsew", padx=(10, 0), pady=5)

        return frame

    def upload_and_analyze_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(file_path)
        self.img_status_lbl.configure(text=f"Analyzing {filename}...")

        result = self.image_proc.process_image(file_bytes, filename, save_incident=True)
        if not result["success"]:
            messagebox.showerror("Analysis Error", result.get("error"))
            self.img_status_lbl.configure(text="Error analyzing image.")
            return

        summary = result["summary"]
        self.img_status_lbl.configure(
            text=f"Status: {summary['status']} | Fire: {summary['fire_count']}, Smoke: {summary['smoke_count']} | Peak Conf: {summary['highest_confidence']:.1%}"
        )

        # Display images
        def render_to_label(rgb_arr, label):
            h, w = rgb_arr.shape[:2]
            target_w = 440
            target_h = int(h * (target_w / w))
            resized = cv2.resize(rgb_arr, (target_w, target_h))
            pil_img = Image.fromarray(resized)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            label.configure(image=ctk_img, text="")

        render_to_label(result["original_image"], self.orig_img_display)
        render_to_label(result["annotated_image"], self.annotated_img_display)

    # ==========================================
    # PAGE 3: VIDEO DETECTION
    # ==========================================
    def _create_video_page(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container, fg_color="#111827")
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(frame, text="Video Stream Processor", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F9FAFB")
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.upload_vid_btn = ctk.CTkButton(
            btn_row,
            text="🎞️ Upload Video File (.mp4, .avi, .mov)",
            fg_color="#EC4899",
            hover_color="#DB2777",
            command=self.process_video_dialog
        )
        self.upload_vid_btn.pack(side="left", padx=(0, 10))

        self.vid_status_lbl = ctk.CTkLabel(btn_row, text="No video selected", text_color="#9CA3AF")
        self.vid_status_lbl.pack(side="left")

        self.vid_progress = ctk.CTkProgressBar(frame)
        self.vid_progress.grid(row=2, column=0, sticky="ew", padx=20, pady=(50, 0))
        self.vid_progress.set(0)

        self.vid_summary_box = ctk.CTkTextbox(frame, height=200, fg_color="#1F2937", text_color="#F9FAFB")
        self.vid_summary_box.grid(row=3, column=0, sticky="nsew", pady=20)

        return frame

    def process_video_dialog(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        filename = os.path.basename(file_path)
        self.vid_status_lbl.configure(text=f"Processing {filename}...")
        self.vid_progress.set(0)
        self.vid_summary_box.delete("1.0", "end")

        def run_proc():
            def cb(pct, text):
                self.after(0, lambda: self.vid_progress.set(pct))
                self.after(0, lambda: self.vid_status_lbl.configure(text=text))

            res = self.video_proc.process_video(
                input_video_path=file_path,
                filename=filename,
                progress_callback=cb,
                save_incident_snapshots=True
            )

            def finish():
                if res["success"]:
                    self.vid_status_lbl.configure(text="✅ Processing Complete!")
                    self.vid_summary_box.insert(
                        "end",
                        f"Status: {res['status']}\n"
                        f"Analyzed Frames: {res['analyzed_frames']}/{res['total_frames']}\n"
                        f"Fire Incidents: {res['fire_detections_count']}\n"
                        f"Smoke Incidents: {res['smoke_detections_count']}\n"
                        f"Speed: {res['average_fps']} FPS\n"
                        f"Annotated Video Saved: {res['output_video_path']}\n"
                    )
                else:
                    self.vid_status_lbl.configure(text="Error processing video.")
                    messagebox.showerror("Video Error", res.get("error"))

            self.after(0, finish)

        threading.Thread(target=run_proc, daemon=True).start()

    # ==========================================
    # PAGE 4: INCIDENT HISTORY
    # ==========================================
    def _create_history_page(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container, fg_color="#111827")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(frame, text="Forensic Incident History", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F9FAFB")
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Table Box
        self.history_box = ctk.CTkTextbox(frame, fg_color="#1F2937", text_color="#F9FAFB", font=ctk.CTkFont(family="Courier", size=12))
        self.history_box.grid(row=1, column=0, sticky="nsew", pady=5)

        # Refresh / Export buttons
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        ctk.CTkButton(btn_row, text="🔄 Refresh Events", fg_color="#3B82F6", command=self.load_history).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_row, text="📥 Export CSV", fg_color="#10B981", command=self.export_csv).pack(side="left")

        return frame

    def load_history(self):
        self.history_box.delete("1.0", "end")
        events = self.db.get_events(limit=100)
        if not events:
            self.history_box.insert("end", "No detection events recorded in database yet.\n")
            return

        header_line = f"{'ID':<6}{'Timestamp':<22}{'Source':<16}{'Class':<18}{'Confidence':<12}{'Snapshot'}\n"
        self.history_box.insert("end", header_line)
        self.history_box.insert("end", "-" * 85 + "\n")

        for ev in events:
            has_snap = "YES" if ev.get("snapshot_path") and os.path.exists(ev["snapshot_path"]) else "NO"
            line = f"{ev['event_id']:<6}{ev['timestamp']:<22}{ev['source']:<16}{ev['detected_class'].upper():<18}{ev['confidence']:<12.1%}{has_snap}\n"
            self.history_box.insert("end", line)

    def export_csv(self):
        export_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="fire_smoke_incidents.csv"
        )
        if export_file:
            self.db.export_to_csv(export_file)
            messagebox.showinfo("Export Successful", f"Saved incident log to:\n{export_file}")

    # ==========================================
    # PAGE 5: SETTINGS
    # ==========================================
    def _create_settings_page(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.main_container, fg_color="#111827")
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(frame, text="System Settings & Diagnostics", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F9FAFB")
        header.grid(row=0, column=0, sticky="w", pady=(0, 15))

        # Model Weights Path
        ctk.CTkLabel(frame, text="YOLO Model Weights Path (.pt)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.model_path_entry = ctk.CTkEntry(frame, width=500, fg_color="#1F2937")
        self.model_path_entry.insert(0, self.detector.model_path)
        self.model_path_entry.grid(row=2, column=0, sticky="w", pady=(0, 10))

        # Simulation Mode
        self.sim_mode_chk = ctk.CTkCheckBox(
            frame,
            text="Enable Simulated Mode (Test UI/Audio/Snapshots without Weights)",
            font=ctk.CTkFont(size=13),
            command=self.toggle_simulation_mode
        )
        if self.detector.is_simulation_mode:
            self.sim_mode_chk.select()
        self.sim_mode_chk.grid(row=3, column=0, sticky="w", pady=10)

        # Audio Toggle
        self.audio_chk = ctk.CTkCheckBox(
            frame,
            text="Enable Local Audio Warning Beeps",
            font=ctk.CTkFont(size=13),
            command=self.toggle_audio
        )
        if self.alerts.audio_enabled:
            self.audio_chk.select()
        self.audio_chk.grid(row=4, column=0, sticky="w", pady=10)

        # Save Button
        ctk.CTkButton(frame, text="Save & Reload Model", fg_color="#10B981", command=self.apply_settings).grid(row=5, column=0, sticky="w", pady=20)

        return frame

    def toggle_simulation_mode(self):
        self.detector.is_simulation_mode = bool(self.sim_mode_chk.get())

    def toggle_audio(self):
        self.alerts.audio_enabled = bool(self.audio_chk.get())

    def apply_settings(self):
        new_path = self.model_path_entry.get().strip()
        self.detector.model_path = new_path
        success, msg = self.detector.load_model()
        if success:
            messagebox.showinfo("Success", "Model loaded successfully!")
        else:
            messagebox.showwarning("Model Warning", msg)

    # Navigation Actions
    def show_live_camera(self):
        self._select_nav("📹 Live Camera")
        self._hide_all_pages()
        self.pages["live"].grid(row=0, column=0, sticky="nsew")

    def show_image_detection(self):
        self._select_nav("🖼️ Image Detection")
        self._hide_all_pages()
        self.pages["image"].grid(row=0, column=0, sticky="nsew")

    def show_video_detection(self):
        self._select_nav("🎥 Video Detection")
        self._hide_all_pages()
        self.pages["video"].grid(row=0, column=0, sticky="nsew")

    def show_history(self):
        self._select_nav("📜 Incident History")
        self._hide_all_pages()
        self.pages["history"].grid(row=0, column=0, sticky="nsew")
        self.load_history()

    def show_settings(self):
        self._select_nav("⚙️ Settings")
        self._hide_all_pages()
        self.pages["settings"].grid(row=0, column=0, sticky="nsew")

    def on_close(self):
        self.is_camera_running = False
        self.camera.stop()
        self.destroy()


if __name__ == "__main__":
    app = FireSmokeApp()
    app.mainloop()
