#!/usr/bin/env python3
"""
Prepare YOLO dataset from cleaned data, fixing path duplication issues.
"""
import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_all_images(input_dir: Path) -> List[Path]:
    """Find all JPG images in the input directory recursively."""
    return list(input_dir.glob('**/*.jpg'))

def find_matching_label(img_path: Path, input_dir: Path) -> Path:
    """
    Find the corresponding label file for an image using advanced matching.
    
    Strategy:
    1. Try direct path replacement (images → labels)
    2. Try searching by filename (ignoring directory structure)
    3. Try searching with different casing (.jpg → .txt)
    """
    # Try direct path match (standard YOLO structure)
    label_path1 = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
    if label_path1.exists():
        return label_path1
    
    # Handle duplication in paths - try searching by name only
    img_name = img_path.stem
    potential_labels = list(input_dir.glob(f'**/{img_name}.txt'))
    
    # If we have labels, return the first one
    if potential_labels:
        return potential_labels[0]
    
    # Try with case variations
    potential_labels = list(input_dir.glob(f'**/{img_name.lower()}.txt')) + list(input_dir.glob(f'**/{img_name.upper()}.txt'))
    if potential_labels:
        return potential_labels[0]
    
    # No label found
    logger.warning(f"No label found for {img_path}")
    return None

def prepare_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Prepare a clean YOLO dataset from the messy structured data.
    
    Args:
        input_dir: Path to input directory with cleaned images/labels
        output_dir: Path to output directory for clean YOLO dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(output_dir / split / 'images', exist_ok=True)
        os.makedirs(output_dir / split / 'labels', exist_ok=True)
    
    # Get all images
    all_images = find_all_images(input_dir)
    logger.info(f"Found {len(all_images)} images in {input_dir}")
    
    # Shuffle and split
    random.shuffle(all_images)
    total = len(all_images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size+val_size]
    test_images = all_images[train_size+val_size:]
    
    # Keep track of stats
    stats = {
        'train': {'images': 0, 'labels': 0},
        'val': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0}
    }
    
    # Process each split
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        logger.info(f"Processing {len(images)} images for {split} set")
        
        for img_path in images:
            # Copy image to output directory
            img_name = img_path.name
            output_img_path = output_dir / split / 'images' / img_name
            
            try:
                shutil.copy2(img_path, output_img_path)
                stats[split]['images'] += 1
                
                # Find and copy corresponding label
                label_path = find_matching_label(img_path, input_dir)
                if label_path and label_path.exists():
                    output_label_path = output_dir / split / 'labels' / label_path.name
                    shutil.copy2(label_path, output_label_path)
                    stats[split]['labels'] += 1
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
    
    # Print summary
    for split in ['train', 'val', 'test']:
        logger.info(f"{split.capitalize()}: {stats[split]['images']} images, {stats[split]['labels']} labels")
    
    # Generate dataset.yaml for YOLOv8
    with open(output_dir / 'dataset.yaml', 'w') as f:
        # First analyze labels to find classes
        class_ids = set()
        for split in ['train', 'val', 'test']:
            label_dir = output_dir / split / 'labels'
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                class_ids.add(int(parts[0]))
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")
        
        if not class_ids:
            class_ids = set(range(10))  # Fallback to 10 classes
        
        # Generate YAML content
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(class_ids)}\n")
        f.write("names:\n")
        for class_id in sorted(list(class_ids)):
            f.write(f"  {class_id}: vehicle_{class_id}\n")
    
    logger.info(f"YOLO dataset prepared at {output_dir}")
    logger.info(f"Created dataset.yaml with {len(class_ids)} classes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from cleaned data, fixing path issues")
    parser.add_argument("--input", required=True, type=str, help="Input directory with cleaned data")
    parser.add_argument("--output", required=True, type=str, help="Output directory for YOLO dataset")
    parser.add_argument("--train", default=0.7, type=float, help="Train ratio")
    parser.add_argument("--val", default=0.2, type=float, help="Validation ratio")
    parser.add_argument("--test", default=0.1, type=float, help="Test ratio")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    args = parser.parse_args()
    
    prepare_yolo_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
