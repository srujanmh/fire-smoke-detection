"""
AegisVision AI — Real-Time Data Ingestion Utility.
Allows you to feed real-time images, video files, or camera captures
into the training/validation dataset and automatically updates data.yaml.
"""

import os
import sys
import shutil
import cv2
from pathlib import Path
from typing import Optional

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def add_image_sample(
    image_path: str,
    label_path: Optional[str] = None,
    split: str = "train"
):
    """Add a single real-time image and optional YOLO annotation to the dataset."""
    src_img = Path(image_path)
    if not src_img.exists():
        print(f"[ERROR] Image file not found: {src_img}")
        return False

    dest_img_dir = DATA_DIR / "images" / split
    dest_lbl_dir = DATA_DIR / "labels" / split
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    dest_img_path = dest_img_dir / src_img.name
    shutil.copy2(src_img, dest_img_path)
    print(f"[INFO] Copied image to: {dest_img_path}")

    if label_path and os.path.exists(label_path):
        src_lbl = Path(label_path)
        dest_lbl_path = dest_lbl_dir / f"{src_img.stem}.txt"
        shutil.copy2(src_lbl, dest_lbl_path)
        print(f"[INFO] Copied label to: {dest_lbl_path}")
    else:
        # Create empty annotation file if background image
        empty_lbl_path = dest_lbl_dir / f"{src_img.stem}.txt"
        empty_lbl_path.touch()
        print(f"[INFO] Created background label: {empty_lbl_path}")

    return True


def extract_frames_from_video(
    video_path: str,
    output_split: str = "train",
    frame_step: int = 15,
    max_frames: int = 100
):
    """Extract sample frames from a real-time recorded video into the dataset."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {video_path}")
        return 0

    dest_img_dir = DATA_DIR / "images" / output_split
    dest_lbl_dir = DATA_DIR / "labels" / output_split
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break

        if frame_idx % frame_step == 0:
            out_name = f"{video_stem}_f{frame_idx:05d}.jpg"
            out_img = dest_img_dir / out_name
            cv2.imwrite(str(out_img), frame)
            # Create corresponding empty label file (to be annotated or kept as background)
            (dest_lbl_dir / f"{video_stem}_f{frame_idx:05d}.txt").touch()
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved_count} frames from {video_path} into {dest_img_dir}")
    return saved_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  1. Add an image: py -3.11 ingest_data.py image path/to/sample.jpg [path/to/sample.txt] [train|val|test]")
        print("  2. Extract from video: py -3.11 ingest_data.py video path/to/video.mp4 [frame_step] [train|val]\n")
        sys.exit(0)

    mode = sys.argv[1].lower()
    if mode == "image":
        img = sys.argv[2]
        lbl = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith('.txt') else None
        split = sys.argv[-1] if sys.argv[-1] in ["train", "val", "test"] else "train"
        add_image_sample(img, lbl, split)
    elif mode == "video":
        vid = sys.argv[2]
        step = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 15
        split = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in ["train", "val"] else "train"
        extract_frames_from_video(vid, output_split=split, frame_step=step)
