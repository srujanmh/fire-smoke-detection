# Model Directory Guide

This directory holds YOLO weights (`.pt` files) trained to detect **Fire** and **Smoke**.

## Expected Model Location

Place your trained YOLO model file here:
```
models/fire_smoke.pt
```

## Supported Model Formats
- Ultralytics YOLOv8 / YOLOv9 / YOLOv11 PyTorch checkpoint files (`.pt`)
- The model must have class names containing terms like `fire` / `flame` and `smoke`.
- The system automatically inspects `model.names` and dynamically maps class IDs to canonical categories (`fire`, `smoke`). It does **not** assume hardcoded class IDs (e.g., class 0 = fire, class 1 = smoke is common, but 0 = smoke, 1 = fire or custom names are seamlessly parsed).

## Obtaining a Fire & Smoke Model

### Option 1: Train your own with YOLOv8 (Recommended for Academic Final Project)
Use the included **Dataset & Training** tab in the dashboard or run the train script:
```bash
python -m modules.detector --train --data data/data.yaml --epochs 50 --imgsz 640
```

### Option 2: Use an existing pretrained checkpoint
If you have trained weights from Google Colab, Kaggle, or Roboflow (e.g. `best.pt`), rename it to `fire_smoke.pt` and place it in this directory:
```bash
copy path\to\your\best.pt models\fire_smoke.pt
```

## Model Safety Guarantee
- The application will **never** silently substitute an unrelated general COCO model (like standard YOLOv8n with 80 everyday objects) and claim it detects fire.
- If no model is found, the application clearly displays a **Model Missing** status with guidance on how to provide or train one.
- A **Simulated Mode** can be explicitly toggled in Settings strictly for testing UI/UX and audio/visual alerts without AI inference; it is prominently labeled as `[SIMULATED - NON-AI]` across all screens.
