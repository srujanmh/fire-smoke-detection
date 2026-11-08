"""
Auto-download model weights on Streamlit Cloud if not present locally.
Set MODEL_URL environment variable to your hosted .pt file (Google Drive, HuggingFace, etc.)
"""
import os
from pathlib import Path

MODEL_PATH = Path("models/fire_smoke.pt")
MODEL_URL = os.environ.get("MODEL_URL", "")

def ensure_model():
    if MODEL_PATH.exists():
        return True
    if not MODEL_URL:
        print("MODEL_URL not set — running in simulation mode.")
        return False
    try:
        import requests
        print(f"Downloading model from {MODEL_URL} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model saved to {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Model download failed: {e} — running in simulation mode.")
        return False

if __name__ == "__main__":
    ensure_model()
