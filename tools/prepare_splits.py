#!/usr/bin/env python3
"""
Prepare train/validation/test splits from cleaned dataset.
"""
import os
import random
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_paths(dataset_dir: Path) -> List[Path]:
    """Get all image paths from the dataset directory."""
    return list(dataset_dir.glob('**/*.jpg'))

def create_split_dirs(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Create train, validation, and test directories."""
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    
    # Create image and label directories for each split
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
    return train_dir, val_dir, test_dir

def copy_with_labels(images: List[Path], dest_dir: Path, base_dir: Path) -> None:
    """Copy images and their corresponding labels to the destination directory."""
    for img_path in images:
        # Get relative path to maintain dataset structure if needed
        rel_path = img_path.relative_to(base_dir)
        label_path = img_path.with_suffix('.txt')
        label_rel_path = label_path.relative_to(base_dir)
        
        # Create destination paths
        dest_img_path = dest_dir / 'images' / rel_path.name
        dest_label_path = dest_dir / 'labels' / label_rel_path.name
        
        # Copy files
        shutil.copy2(img_path, dest_img_path)
        if label_path.exists():
            shutil.copy2(label_path, dest_label_path)
        else:
            logger.warning(f"Label file not found for {img_path}")

def prepare_splits(
    dataset_dir: Path, 
    output_dir: Path, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Prepare train/validation/test splits from cleaned dataset.
    
    Args:
        dataset_dir: Path to the cleaned dataset directory.
        output_dir: Path to output the splits.
        train_ratio: Proportion of dataset for training.
        val_ratio: Proportion of dataset for validation.
        test_ratio: Proportion of dataset for testing.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Verify ratios sum to 1.0
    if not 0.999 <= train_ratio + val_ratio + test_ratio <= 1.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Get all image paths
    all_images = get_image_paths(dataset_dir)
    random.shuffle(all_images)
    total_images = len(all_images)
    
    logger.info(f"Found {total_images} images in {dataset_dir}")
    
    # Calculate split sizes
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size
    
    # Create split subsets
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    logger.info(f"Split sizes - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Create output directories
    train_dir, val_dir, test_dir = create_split_dirs(output_dir)
    
    # Copy files
    logger.info("Copying training images and labels...")
    copy_with_labels(train_images, train_dir, dataset_dir)
    
    logger.info("Copying validation images and labels...")
    copy_with_labels(val_images, val_dir, dataset_dir)
    
    logger.info("Copying test images and labels...")
    copy_with_labels(test_images, test_dir, dataset_dir)
    
    # Print summary
    logger.info(f"Data splits completed - Train: {len(train_images)} ({train_ratio*100:.1f}%), "
                f"Val: {len(val_images)} ({val_ratio*100:.1f}%), "
                f"Test: {len(test_images)} ({test_ratio*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/validation/test splits from cleaned dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to cleaned dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion for training set")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proportion for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Proportion for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_splits(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
