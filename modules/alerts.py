"""
Alert management system for Fire & Smoke Detection.
Supports visual badges, temporal confirmation windows,
audio alert cooldowns, and silence controls.
"""

import time
from collections import deque
from typing import Dict, Any, Optional
from utils.logging_config import logger

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlertSystem:
    """
    Manages detection alerts with temporal confirmation and cooldowns
    to eliminate frame flicker and audio buzzing.
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        temporal_window_size: int = 5,
        temporal_min_detections: int = 3,
        audio_enabled: bool = True
    ):
        self.cooldown_seconds = cooldown_seconds
        self.temporal_window_size = temporal_window_size
        self.temporal_min_detections = temporal_min_detections
        self.audio_enabled = audio_enabled

        # Sliding window of boolean flags [True if detection in frame, False otherwise]
        self.fire_window = deque(maxlen=temporal_window_size)
        self.smoke_window = deque(maxlen=temporal_window_size)

        self.last_fire_alert_time: float = 0.0
        self.last_smoke_alert_time: float = 0.0
        self.current_alert_state: str = "NORMAL"
        self.silenced: bool = False

    def reset_state(self) -> None:
        """Clear temporal history and reset current alert state."""
        self.fire_window.clear()
        self.smoke_window.clear()
        self.current_alert_state = "NORMAL"
        logger.info("Alert state and temporal buffers reset.")

    def update_temporal_window(self, new_size: int, new_min_detections: int) -> None:
        """
        Resize the sliding window buffers at runtime.
        Python deques have an immutable maxlen, so we rebuild them while
        preserving recent history up to the new size.
        """
        new_size = max(1, int(new_size))
        new_min_detections = max(1, min(int(new_min_detections), new_size))
        self.fire_window = deque(list(self.fire_window)[-new_size:], maxlen=new_size)
        self.smoke_window = deque(list(self.smoke_window)[-new_size:], maxlen=new_size)
        self.temporal_window_size = new_size
        self.temporal_min_detections = new_min_detections
        logger.info(
            f"Temporal window updated: size={new_size}, min_detections={new_min_detections}"
        )

    def set_silenced(self, silenced: bool) -> None:
        """Mute or unmute audio notifications."""
        self.silenced = silenced

    def process_frame_detections(
        self,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process single frame detection result, update temporal window,
        and evaluate whether a confirmed alert should be raised.
        """
        fire_count = detection_result.get("fire_count", 0)
        smoke_count = detection_result.get("smoke_count", 0)

        # Update sliding windows
        self.fire_window.append(1 if fire_count > 0 else 0)
        self.smoke_window.append(1 if smoke_count > 0 else 0)

        # Temporal confirmation: require at least M positive frames out of N
        confirmed_fire = sum(self.fire_window) >= self.temporal_min_detections
        confirmed_smoke = sum(self.smoke_window) >= self.temporal_min_detections

        # Determine consolidated alert state
        if confirmed_fire and confirmed_smoke:
            state = "FIRE AND SMOKE DETECTED"
        elif confirmed_fire:
            state = "FIRE DETECTED"
        elif confirmed_smoke:
            state = "SMOKE DETECTED"
        elif fire_count > 0 or smoke_count > 0:
            # Detected in current frame but not yet temporally confirmed across window
            state = "UNCERTAIN"
        else:
            state = "NORMAL"

        self.current_alert_state = state
        now = time.time()

        # Audio trigger check
        should_play_sound = False
        alert_type = None

        if not self.silenced and self.audio_enabled:
            if confirmed_fire and (now - self.last_fire_alert_time) > self.cooldown_seconds:
                should_play_sound = True
                alert_type = "FIRE"
                self.last_fire_alert_time = now
            elif confirmed_smoke and (now - self.last_smoke_alert_time) > self.cooldown_seconds:
                should_play_sound = True
                alert_type = "SMOKE"
                self.last_smoke_alert_time = now

        if should_play_sound:
            self._trigger_local_beep(alert_type)

        return {
            "state": state,
            "confirmed_fire": confirmed_fire,
            "confirmed_smoke": confirmed_smoke,
            "fire_buffer_count": sum(self.fire_window),
            "smoke_buffer_count": sum(self.smoke_window),
            "buffer_capacity": self.temporal_window_size,
            "should_play_sound": should_play_sound,
            "alert_type": alert_type,
            "silenced": self.silenced
        }

    def _trigger_local_beep(self, alert_type: Optional[str]) -> None:
        """Trigger local Windows beep if available, non-blocking fallback."""
        if WINSOUND_AVAILABLE:
            try:
                frequency = 1200 if alert_type == "FIRE" else 800
                duration_ms = 250
                # Use winsound async flag where possible or short beep
                winsound.Beep(frequency, duration_ms)
            except Exception as e:
                logger.debug(f"Local beep suppressed or unavailable: {e}")

    @staticmethod
    def get_web_audio_html(state: str) -> str:
        """
        Generate lightweight HTML/JS Web Audio API synthesizer for browser playback.
        Works across all operating systems inside Streamlit without external wav files.
        """
        freq = 880 if "FIRE" in state else 520
        return f"""
        <script>
        (function() {{
            try {{
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime({freq}, audioCtx.currentTime);
                gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.4);
            }} catch(e) {{
                console.log("Audio alert suppressed:", e);
            }}
        }})();
        </script>
        """
