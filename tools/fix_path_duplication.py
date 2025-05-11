#!/usr/bin/env python3
"""
Fix path duplication in dataset directory structure.
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_duplicated_paths(root_dir: Path) -> List[Path]:
    """Find paths that have duplicated directory structures."""
    duplicated_paths = []
    for img_path in root_dir.glob('**/*.jpg'):
        parts = str(img_path).split(os.sep)
        # Check for patterns like '1/folder/1/folder' or pattern repetition
        for i in range(len(parts) - 1):
            remaining = len(parts) - i
            if remaining >= 4:  # Need at least 4 parts to have a duplication
                pattern = parts[i:i+2]
                if pattern == parts[i+2:i+4]:
                    duplicated_paths.append(img_path)
                    break
    return duplicated_paths

def get_corrected_path(path: Path, root_dir: Path) -> Optional[Path]:
    """Generate a corrected path without duplications."""
    path_str = str(path)
    parts = path_str.split(os.sep)
    
    # Detect patterns of duplication
    cleaned_parts = []
    i = 0
    while i < len(parts):
        cleaned_parts.append(parts[i])
        # Look for duplicated pattern
        pattern_length = 2
        pattern = parts[i:i+pattern_length]
        if i + pattern_length < len(parts) and pattern == parts[i+pattern_length:i+2*pattern_length]:
            # Skip the duplicated pattern
            i += pattern_length
        else:
            i += 1
            
    # Reconstruct path
    cleaned_path = os.sep.join(cleaned_parts)
    
    # Make sure we haven't lost the root path
    if not cleaned_path.startswith(str(root_dir)):
        logger.warning(f"Path fix failed for {path}, returned path doesn't start with root: {cleaned_path}")
        return None
        
    return Path(cleaned_path)

def fix_duplicated_paths(root_dir: Path, output_dir: Path = None) -> int:
    """
    Fix duplicated directory structures in dataset.
    
    Args:
        root_dir: Path to the dataset directory with duplicated paths.
        output_dir: Path to output fixed structure. If None, modify in place.
        
    Returns:
        Number of fixed paths.
    """
    # If no output dir specified, we'll modify in place
    in_place = output_dir is None
    if in_place:
        output_dir = root_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image paths with duplications
    logger.info(f"Scanning for duplicated paths in {root_dir}...")
    duplicated_paths = find_duplicated_paths(root_dir)
    
    if not duplicated_paths:
        logger.info("No duplicated paths found.")
        return 0
    
    logger.info(f"Found {len(duplicated_paths)} paths with duplication.")
    fixed_count = 0
    
    # Process each duplicated path
    for old_path in duplicated_paths:
        # Get the corrected path
        new_path = get_corrected_path(old_path, root_dir)
        if new_path is None:
            logger.warning(f"Could not fix path: {old_path}")
            continue
            
        # If output_dir is different from root_dir, adjust the new_path
        if not in_place:
            new_path = output_dir / new_path.relative_to(root_dir)
        
        # Create directories if needed
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy/move the image file
        if old_path != new_path:
            if in_place:
                logger.info(f"Moving {old_path} -> {new_path}")
                # For in-place, we move rather than copy
                try:
                    old_path.rename(new_path)
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Failed to move {old_path}: {e}")
            else:
                logger.info(f"Copying {old_path} -> {new_path}")
                try:
                    shutil.copy2(old_path, new_path)
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Failed to copy {old_path}: {e}")
        
        # Check for corresponding label file
        old_label_path = old_path.with_suffix('.txt')
        if old_label_path.exists():
            new_label_path = new_path.with_suffix('.txt')
            
            if in_place:
                if old_label_path != new_label_path:
                    try:
                        old_label_path.rename(new_label_path)
                    except Exception as e:
                        logger.error(f"Failed to move label file {old_label_path}: {e}")
            else:
                try:
                    shutil.copy2(old_label_path, new_label_path)
                except Exception as e:
                    logger.error(f"Failed to copy label file {old_label_path}: {e}")
    
    logger.info(f"Fixed {fixed_count} duplicated paths.")
    return fixed_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix duplicated directory structures in dataset")
    parser.add_argument("--input", type=str, required=True, help="Input directory with duplicated paths")
    parser.add_argument("--output", type=str, default=None, help="Output directory for fixed structure (if not specified, modify in place)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    fix_duplicated_paths(input_dir, output_dir)
