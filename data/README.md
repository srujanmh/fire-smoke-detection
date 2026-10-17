# Dataset Preparation Guide

This directory holds ground-truth datasets for training and evaluating the Fire & Smoke detection model.

## Standard YOLO Dataset Structure

The application expects the standard YOLO directory layout:

```
data/
├── data.yaml
├── images/
│   ├── train/       # Training images (.jpg, .jpeg, .png)
│   ├── val/         # Validation images
│   └── test/        # Test images for empirical evaluation
└── labels/
    ├── train/       # YOLO format annotations (.txt)
    ├── val/
    └── test/
```

## Annotation Format (YOLO Text Format)
Each image file in `images/<split>/<filename>.jpg` has a corresponding label file in `labels/<split>/<filename>.txt`.

Each line in the `.txt` file represents one bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized floating-point numbers between `0.0` and `1.0`:
- `x_center`: horizontal center coordinate of the box divided by image width.
- `y_center`: vertical center coordinate of the box divided by image height.
- `width`: box width divided by image width.
- `height`: box height divided by image height.

Example `labels/test/frame_001.txt`:
```
0 0.452 0.512 0.220 0.310
1 0.612 0.380 0.350 0.420
```

## `data.yaml` Specification

```yaml
path: ../data  # dataset root dir (or absolute path)
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 2

# Class names
names: ['fire', 'smoke']
```

## Dataset Integrity Checks in Dashboard
Navigate to the **Dataset & Training** page to run automated validation:
1. Missing image or label pairs.
2. Coordinates outside `[0.0, 1.0]`.
3. Degenerate boxes (`width <= 0` or `height <= 0`).
4. Class balance statistics (fire vs. smoke count).
