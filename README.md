# Data Cleanup Pipeline for Vehicle Detection Datasets

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](https://github.com/PaulrydrickPuri/data-cleanup/blob/main/LICENSE)
[![For: CoreframeAI](https://img.shields.io/badge/For-CoreframeAI-blue.svg)](https://github.com/PaulrydrickPuri/data-cleanup)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/PaulrydrickPuri/data-cleanup)

A comprehensive preprocessing system for cleaning, validating, and optimizing vehicle detection datasets before model training.

> **IMPORTANT**: This repository is proprietary and for exclusive use by CoreframeAI. Unauthorized use, distribution, or modification is strictly prohibited.

## Overview

This pipeline implements a fail-fast preprocessing system that:

1. Uses gold-standard (hand-verified) images as class anchors
2. Filters noisy images for wrong labels, low quality, and feature mismatch
3. Ensures only high-quality data enters the training pipeline

## Key Features

- Object detection using YOLOv8
- Instance segmentation using Segment Anything Model (SAM)
- Class validation using CLIP embeddings (with selective validation)
- OCR verification for license plates
- Quality filtering (blur, size, exposure)
- COCO format conversion
- Comprehensive audit trail
- Optimized performance with persistent model instances

## Pipeline Flow

1. **Object Detection (YOLOv8)**
   - Identifies vehicles in images
   - Provides initial bounding box coordinates
   - Assigns class predictions

2. **Segmentation (SAM)**
   - Processes each detected bounding box
   - Creates precise masks for each object
   - Filters out boxes with too much background
   - Ensures only dominant objects proceed to validation

3. **Class Validation (CLIP)**
   - Takes filtered bounding boxes from SAM
   - Compares object crops against class embeddings
   - Validates if detected class matches visual features
   - Selected images only (performance optimization)

4. **OCR Check**
   - Identifies text in images (license plates)
   - Verifies against regex patterns
   - Flags objects with text for privacy concerns
   - Selected images only (performance optimization)

5. **Quality Filtering**
   - Applies blur, size, and exposure thresholds
   - Ensures only high-quality objects are included
   - Final validation before dataset output

## Repository Structure

```
data-cleanup/
├── config/                  # Configuration files
│   ├── .env                 # Environment variables
│   ├── env_config.txt       # Configuration settings
│   └── dataset_fixed.yaml   # YOLO dataset configuration
│
├── data/                    # All data-related directories
│   ├── Dataset/             # Original input dataset
│   ├── cleaned_datasets/    # Pipeline output with optimized processing
│   ├── cropped_vehicles/    # Vehicle crops from detection
│   └── [other data dirs]    # Various dataset directories
│
├── data_cleanup/            # Core pipeline implementation
│   ├── clean_dataset.py     # Main cleanup implementation
│   ├── clean_dataset_single.py  # Optimized single-image processing
│   ├── detect_objects.py    # Object detection module
│   ├── segment_masks.py     # SAM segmentation module
│   ├── validate_class.py    # CLIP validation module
│   ├── ocr_check.py         # License plate OCR verification
│   ├── quality_filter.py    # Image quality filtering
│   ├── convert_format.py    # Format conversion utilities
│   └── assets/              # Configuration files
│       ├── gt_anchors.json  # CLIP embeddings for class prototypes
│       └── regex_plate.json # License plate regex patterns
│
├── logs/                    # Log files and audit records
│   ├── data_cleanup.log     # Pipeline log files
│   └── data_cleanup_audit.json  # Audit trail from processing
│
├── models/                  # Model weights and training outputs
│   ├── outputs/             # YOLOv8 training results
│   ├── sam_vit_h_4b8939.pth # SAM model weights
│   └── yolov8s.pt           # YOLO model weights
│
├── reports/                 # Documentation and reports
│   └── optimization_report_2025-05-11.md  # Pipeline performance report
│
├── scripts/                 # Executable scripts
│   ├── pause_resume_cleanup.py  # Main entry point with pause/resume
│   ├── train_yolov8.py      # YOLOv8 training script
│   └── [other scripts]      # Various utility scripts
│
├── tools/                   # Utilities and helpers
│   ├── visualize_results.py  # Results visualization tool
│   ├── fix_path_duplication.py  # Path handling fixes
│   └── prepare_yolo_dataset.py  # YOLO dataset preparation
│
└── viewers/                 # Visual inspection tools
    └── gt_viewer.html       # Ground truth visualization
```

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Performance Optimizations

This pipeline includes several key optimizations to improve processing speed while maintaining quality:

1. **Persistent Model Instances**
   - CLIP and OCR models are loaded only once and kept in memory
   - Eliminates repeated loading overhead for each image
   - Reduces processing time by ~60%

2. **Selective Validation**
   - Class validation runs on every 10th image by default
   - Preserves quality control while significantly reducing processing time
   - Can be adjusted via environment variables

3. **Efficient Data Flow**
   - Optimized pipeline ensures data flows correctly between stages
   - Each stage only processes necessary information
   - Only filtered bounding boxes proceed to validation steps

These optimizations have reduced processing time from 20+ hours to under 10 hours for a 4000+ image dataset while maintaining 90% data retention rate.

## Usage

### 1. Generate Anchor Embeddings

First, generate anchor embeddings from your ground truth images:

```bash
python -m data_cleanup.validate_class generate \
  --images data/gt_for_embeddings/gt_images.txt \
  --labels data/gt_for_embeddings/gt_labels.txt \
  --output data_cleanup/assets/gt_anchors.json
```

Where:
- `gt_images.txt`: Text file with paths to ground truth images (one per line)
- `gt_labels.txt`: Text file with class labels (one per line)

### 2. Clean Dataset with Optimized Pipeline

Run the optimized pipeline with pause/resume functionality:

```bash
python scripts/pause_resume_cleanup.py \
  --dataset data/Dataset \
  --output_dir data/cleaned_datasets/output \
  --anchors data_cleanup/assets/gt_anchors.json \
  --regex data_cleanup/assets/regex_plate.json \
  --confidence 0.25 \
  --similarity 0.65 \
  --blur 100.0 \
  --min-size 64 \
  --min-exposure 30 \
  --max-exposure 225
```

### 3. Prepare Dataset for Training

Prepare the cleaned dataset for model training with proper train/val/test splits:

```bash
python tools/prepare_yolo_dataset.py \
  --input data/cleaned_datasets/output \
  --output data/yolo-ready \
  --train 0.7 \
  --val 0.2 \
  --test 0.1
```

### 4. Train YOLOv8 Model

Train a YOLOv8 model on the prepared dataset:

```bash
python scripts/train_yolov8.py \
  --dataset data/yolo-ready \
  --output models/outputs/training \
  --model s \
  --epochs 100 \
  --batch 16
```

### 5. Visualize Results

Visualize the model predictions and dataset:

```bash
python tools/visualize_results.py \
  --dataset data/yolo-ready \
  --predictions models/outputs/training/predictions
```

### 6. Convert to COCO Format (Optional)

Convert the cleaned dataset to COCO format if needed:

```bash
python -m data_cleanup.convert_format \
  --images data/yolo-ready/train/images \
  --labels data/yolo-ready/train/labels \
  --output data/yolo-ready/instances_train.json
```

## Environment Variables

Create a `.env` file with the following options:

```
LOG_JSON=true                # Enable JSON structured logging
LOG_RAW_CROPS=false          # Disable saving rejected crops (for privacy)
```

## Testing

Run the test suite:

```bash
pytest data_cleanup/tests/
```

## Ethical Considerations

- The pipeline stores only hashes of license plate text, never raw PII
- Discarded images are logged by path only, not copied
- Toggle `LOG_RAW_CROPS=false` to disable sensitive artifact dumps

## References

1. Kirillov et al., 2023 – "Segment Anything" (CVPR)
2. Radford et al., 2021 – "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, ICML)
