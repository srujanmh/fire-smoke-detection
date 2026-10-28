"""
AegisVision AI — YOLO Model Training Pipeline.
Fine-tunes YOLOv8 on the prepared Fire & Smoke dataset.
Outputs model weights, confusion matrices, loss graphs, and metrics to models/training_runs/.
"""

import sys
import argparse
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO
from utils.logging_config import logger


def train_fire_smoke_model(
    data_yaml: str = "data/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    workers: int = 2,
    project_dir: str = "models/training_runs",
    run_name: str = "fire_smoke_custom",
    save_best_as_default: bool = True
):
    """Run supervised fine-tuning of YOLO on the fire & smoke dataset."""

    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    if not data_path.exists():
        logger.error(f"Dataset config not found at: {data_path}")
        print(f"[ERROR] Dataset configuration file not found at: {data_path}")
        return False

    # Determine hardware device
    if device == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    print("=" * 65)
    print(" 🔥 AegisVision AI — Fire & Smoke Model Training Pipeline")
    print("=" * 65)
    print(f" Dataset YAML : {data_path}")
    print(f" Base Model   : {base_model}")
    print(f" Epochs       : {epochs}")
    print(f" Image Size   : {imgsz}x{imgsz}")
    print(f" Batch Size   : {batch}")
    print(f" Target Device: {dev.upper()}")
    print(f" Output Dir   : {project_dir}/{run_name}")
    print("=" * 65 + "\n")

    logger.info(f"Loading base architecture: {base_model}...")
    model = YOLO(base_model)

    logger.info("Commencing supervised neural network training...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        workers=workers,
        project=str(PROJECT_ROOT / project_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    trained_weights_path = PROJECT_ROOT / project_dir / run_name / "weights" / "best.pt"
    print("\n" + "=" * 65)
    print(" ✅ Training Completed Successfully!")
    print(f" Best Weights Saved to: {trained_weights_path}")
    print("=" * 65)

    if save_best_as_default and trained_weights_path.exists():
        target_path = PROJECT_ROOT / "models" / "fire_smoke.pt"
        import shutil
        shutil.copy2(trained_weights_path, target_path)
        print(f" [INFO] Deployed best weights to default model location: {target_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Fire & Smoke Dataset")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: 'cpu', '0', or 'auto'")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers (default: 2)")
    parser.add_argument("--name", type=str, default="fire_smoke_custom", help="Experiment name")
    parser.add_argument("--no-deploy", action="store_true", help="Do not overwrite models/fire_smoke.pt with trained weights")
    args = parser.parse_args()

    train_fire_smoke_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        run_name=args.name,
        save_best_as_default=not args.no_deploy
    )
