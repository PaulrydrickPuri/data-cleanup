#!/usr/bin/env python3
"""
Train YOLOv8 model on the cleaned vehicle detection dataset.
"""
import os
import yaml
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def create_dataset_yaml(dataset_path: Path, output_path: Path) -> Path:
    """
    Create a dataset.yaml file for YOLOv8 training.
    
    Args:
        dataset_path: Path to the dataset directory.
        output_path: Path to save the YAML file.
        
    Returns:
        Path to the created YAML file.
    """
    # Get class names from labels
    class_ids = set()
    label_files = list(Path(dataset_path / 'train' / 'labels').glob('*.txt'))
    
    if not label_files:
        raise ValueError(f"No label files found in {dataset_path / 'train' / 'labels'}")
        
    logger.info(f"Scanning {len(label_files)} label files to extract class information")
    
    # First pass: collect all class IDs
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:  # YOLO format requires 5+ parts
                        class_id = int(parts[0])
                        class_ids.add(class_id)
        except Exception as e:
            logger.warning(f"Error reading {label_file}: {e}")
    
    if not class_ids:
        # Fallback to using default class IDs if none found
        logger.warning("No valid class IDs found in labels, using default classes 0-9")
        class_ids = set(range(10))
        
    logger.info(f"Found {len(class_ids)} unique class IDs: {sorted(list(class_ids))}")
    
    # Create class name dictionary
    names_dict = {i: f'vehicle_{i}' for i in sorted(list(class_ids))}
    
    yaml_dict = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(names_dict),
        'names': names_dict
    }
    
    logger.info(f"Creating dataset YAML with {len(names_dict)} classes")
    
    
    # Write dataset.yaml
    yaml_file = output_path / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Created dataset YAML at {yaml_file}")
    
    return yaml_file

def train_yolov8(
    dataset_path: Path,
    output_dir: Path,
    model_size: str = 's',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    pretrained: bool = True,
    device: str = '0'  # Use GPU 0 by default
) -> None:
    """
    Train YOLOv8 model on the dataset.
    
    Args:
        dataset_path: Path to the dataset directory.
        output_dir: Path to save the training results.
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x').
        epochs: Number of training epochs.
        batch_size: Batch size.
        img_size: Image size for training.
        pretrained: Whether to use pretrained weights.
        device: Device to train on ('cpu', '0', '0,1', etc.).
    """
    try:
        # Import here to avoid dependency errors if not training
        from ultralytics import YOLO
        
        # Create dataset.yaml
        os.makedirs(output_dir, exist_ok=True)
        dataset_yaml = create_dataset_yaml(dataset_path, output_dir)
        
        # Initialize model
        model_name = f"yolov8{model_size}.pt"
        model = YOLO(model_name)
        
        # Train model
        logger.info(f"Starting YOLOv8{model_size} training for {epochs} epochs")
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=50,
            device=device,
            project=str(output_dir),
            name=f"yolov8{model_size}_vehicle_detection"
        )
        
        # Validate model
        logger.info("Validation on test set:")
        model.val(data=str(dataset_yaml), split='test')
        
        logger.info(f"Training completed. Results saved to {output_dir}")
        
    except ImportError:
        logger.error("YOLOv8 not installed. Please run: pip install ultralytics")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on vehicle detection dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory with train/val/test splits")
    parser.add_argument("--output", type=str, default="outputs/training", help="Output directory for training results")
    parser.add_argument("--model", type=str, default="s", help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device to train on (cpu, 0, 0,1, etc.)")
    
    args = parser.parse_args()
    
    train_yolov8(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )
