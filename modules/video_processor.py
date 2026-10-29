"""
Video processing pipeline for uploaded video files.
Streams frames sequentially to minimize memory footprint,
applies configurable sampling intervals, generates annotated video,
and records incidents with video timestamps.
"""

import os
import cv2
import time
import tempfile
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

from modules.detector import FireSmokeDetector
from modules.event_manager import EventManager
from utils.file_validation import validate_video_file, sanitize_filename
from utils.image_utils import draw_detections, rgb_to_bgr, bgr_to_rgb
from utils.logging_config import logger


class VideoProcessor:
    """Memory-efficient video inference and encoding engine."""

    def __init__(
        self,
        detector: FireSmokeDetector,
        event_manager: EventManager,
        output_dir: str = "outputs/processed_videos"
    ):
        self.detector = detector
        self.event_manager = event_manager
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract frame rate, resolution, frame count, and duration from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": round(duration_sec, 2),
            "error": None
        }

    def process_video(
        self,
        input_video_path: str,
        filename: str,
        frame_interval: int = 2,
        confidence_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        save_incident_snapshots: bool = True
    ) -> Dict[str, Any]:
        """
        Process a video frame-by-frame. Writes annotated video to outputs/processed_videos/.
        """
        clean_filename = sanitize_filename(filename)
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            return {
                "success": False,
                "error": "Could not open video file. Invalid or unsupported video codec.",
                "status": "CODEC_ERROR"
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        # Output video file path
        output_filename = f"processed_{Path(clean_filename).stem}.mp4"
        output_filepath = os.path.join(self.output_dir, output_filename)

        # FourCC codec - try mp4v first
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Effective FPS of output video matches input
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

        if not out.isOpened():
            # Fallback to XVID in case mp4v isn't supported on host OpenCV
            logger.warning("mp4v codec failed, trying XVID...")
            output_filepath = os.path.join(self.output_dir, f"processed_{Path(clean_filename).stem}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

        frame_idx = 0
        analyzed_frames = 0
        fire_events_count = 0
        smoke_events_count = 0
        total_detections_count = 0
        highest_conf_overall = 0.0

        sample_detections: List[Dict[str, Any]] = []
        last_annotated_frame = None
        start_wall_time = time.time()

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                frame_idx += 1
                video_timestamp_sec = round(frame_idx / fps, 2)

                # Process every Nth frame
                if frame_idx % frame_interval == 0:
                    analyzed_frames += 1
                    rgb_frame = bgr_to_rgb(frame_bgr)

                    # Run inference
                    det_result = self.detector.detect(
                        rgb_frame,
                        confidence_threshold=confidence_threshold,
                        is_rgb=True
                    )

                    detections = det_result.get("detections", [])
                    annotated_rgb = draw_detections(rgb_frame, detections, is_rgb=True)
                    annotated_bgr = rgb_to_bgr(annotated_rgb)
                    last_annotated_frame = annotated_rgb

                    if detections:
                        total_detections_count += len(detections)
                        fire_events_count += det_result.get("fire_count", 0)
                        smoke_events_count += det_result.get("smoke_count", 0)
                        highest_conf_overall = max(highest_conf_overall, det_result.get("highest_confidence", 0.0))

                        # Log incident with video timestamp
                        if save_incident_snapshots and det_result.get("status") in ["FIRE DETECTED", "SMOKE DETECTED", "FIRE AND SMOKE DETECTED"]:
                            self.event_manager.process_incident(
                                frame=annotated_rgb,
                                detection_result=det_result,
                                source="video_upload",
                                filename=clean_filename,
                                video_timestamp=video_timestamp_sec
                            )

                        if len(sample_detections) < 10:
                            sample_detections.append({
                                "frame": frame_idx,
                                "timestamp": video_timestamp_sec,
                                "status": det_result.get("status"),
                                "detections": detections
                            })

                    out.write(annotated_bgr)
                else:
                    # For skipped frames, write original frame to maintain duration
                    out.write(frame_bgr)

                # Update progress
                if progress_callback and total_frames > 0:
                    progress = min(1.0, frame_idx / total_frames)
                    progress_callback(
                        progress,
                        f"Analyzing frame {frame_idx}/{total_frames} ({video_timestamp_sec:.1f}s)"
                    )

        except Exception as e:
            logger.error(f"Error processing video stream: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Video processing interrupted: {str(e)}",
                "status": "PROCESSING_ERROR"
            }
        finally:
            cap.release()
            out.release()

        total_wall_time = time.time() - start_wall_time
        avg_fps = analyzed_frames / total_wall_time if total_wall_time > 0 else 0.0

        logger.info(
            f"Processed {analyzed_frames}/{frame_idx} frames from '{clean_filename}' "
            f"in {total_wall_time:.1f}s ({avg_fps:.1f} FPS)."
        )

        overall_status = "NORMAL"
        if fire_events_count > 0 and smoke_events_count > 0:
            overall_status = "FIRE AND SMOKE DETECTED"
        elif fire_events_count > 0:
            overall_status = "FIRE DETECTED"
        elif smoke_events_count > 0:
            overall_status = "SMOKE DETECTED"

        return {
            "success": True,
            "error": None,
            "status": overall_status,
            "output_video_path": output_filepath,
            "total_frames": frame_idx,
            "analyzed_frames": analyzed_frames,
            "fire_detections_count": fire_events_count,
            "smoke_detections_count": smoke_events_count,
            "total_detections_count": total_detections_count,
            "highest_confidence": highest_conf_overall,
            "processing_duration_sec": round(total_wall_time, 2),
            "average_fps": round(avg_fps, 1),
            "sample_detections": sample_detections,
            "last_annotated_frame": last_annotated_frame
        }
