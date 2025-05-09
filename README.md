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
- Class validation using CLIP embeddings
- OCR verification for license plates
- Quality filtering (blur, size, exposure)
- COCO format conversion
- Comprehensive audit trail

## Repository Structure

```
data_cleanup/
├── detect_objects.py     # YOLOv8 object detection
├── segment_masks.py      # SAM instance segmentation
├── validate_class.py     # CLIP embedding validation
├── ocr_check.py          # License plate OCR verification
├── quality_filter.py     # Image quality checks
├── clean_dataset.py      # Pipeline orchestrator
├── convert_format.py     # YOLO to COCO conversion
├── tests/                # Pytest test suite
└── assets/               # Configuration files
    ├── gt_anchors.json   # CLIP embeddings for class prototypes
    └── regex_plate.json  # License plate regex patterns
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

## Usage

### 1. Generate Anchor Embeddings

First, generate anchor embeddings from your ground truth images:

```bash
python -m data_cleanup.validate_class generate \
  --images gt_images.txt \
  --labels gt_labels.txt \
  --output data_cleanup/assets/gt_anchors.json
```

Where:
- `gt_images.txt`: Text file with paths to ground truth images (one per line)
- `gt_labels.txt`: Text file with class labels (one per line)

### 2. Clean Dataset

Run the full pipeline:

```bash
python -m data_cleanup.clean_dataset \
  /path/to/input_dataset \
  /path/to/output_dataset \
  --anchors data_cleanup/assets/gt_anchors.json \
  --regex data_cleanup/assets/regex_plate.json \
  --confidence 0.25 \
  --similarity 0.65 \
  --blur 100.0 \
  --min-size 64 \
  --min-exposure 30 \
  --max-exposure 225
```

### 3. Convert to COCO Format

Convert the cleaned dataset to COCO format:

```bash
python -m data_cleanup.convert_format \
  --images /path/to/output_dataset/images \
  --labels /path/to/output_dataset/labels \
  --masks /path/to/output_dataset/masks \
  --output /path/to/output_dataset/instances_train.json
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
