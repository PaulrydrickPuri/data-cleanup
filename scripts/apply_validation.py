#!/usr/bin/env python3
"""
Script to apply validation steps (CLIP and OCR) to an already cleaned dataset.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
import shutil
from tqdm import tqdm

from data_cleanup.validate_class import validate_class
from data_cleanup.ocr_check import ocr_check

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validation.log")
    ]
)
logger = logging.getLogger(__name__)

def find_images(directory):
    """Find all images in the dataset, excluding mask files."""
    image_paths = []
    for root, _, files in os.walk(directory):
        # Skip mask directories
        if 'masks' in root.lower():
            continue
            
        for file in files:
            # Skip mask files and temporary files
            if ('mask' in file.lower() or 
                file.startswith('.') or 
                file.startswith('~')):
                continue
                
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                # Convert to absolute path for consistent tracking
                image_path = os.path.abspath(image_path)
                image_paths.append(image_path)
    return image_paths

def validate_image(image_path, output_dir, anchors_path, regex_path, similarity_threshold=0.65):
    """
    Apply CLIP validation and OCR check to a single image in the cleaned dataset.
    
    Args:
        image_path: Path to the image file.
        output_dir: Output directory for validated images.
        anchors_path: Path to anchors file for CLIP validation.
        regex_path: Path to regex patterns for OCR check.
        similarity_threshold: Threshold for class similarity.
    """
    image_path = Path(image_path)
    
    # Find corresponding label file
    label_path = image_path.with_suffix('.txt')
    if not label_path.exists():
        # Try looking in a labels directory
        parent_dir = image_path.parent
        if 'images' in str(parent_dir):
            labels_dir = str(parent_dir).replace('images', 'labels')
            label_path = Path(labels_dir) / image_path.with_suffix('.txt').name
    
    if not label_path.exists():
        logger.warning(f"No label file found for {image_path}")
        return False
    
    # Create output directory structure
    rel_path = os.path.relpath(str(image_path), str(args.input_dir))
    output_img_dir = Path(output_dir) / os.path.dirname(rel_path)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Create label directory
    output_label_dir = str(output_img_dir).replace('images', 'labels')
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Read and parse YOLO format labels
    objects = []
    img_width, img_height = None, None
    
    # Get image dimensions
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        img_height, img_width = img.shape[:2]
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        return False
    
    # Read YOLO format label (class_id x_center y_center width height)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Skip if no objects
    if not lines:
        logger.warning(f"No objects in label file: {label_path}")
        return False
        
    # Parse YOLO format labels
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert from YOLO format (normalized) to pixel coordinates
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            # Create object dictionary in the format expected by validation functions
            obj = {
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': 1.0  # Original annotations are treated as 100% confident
            }
            objects.append(obj)
    
    try:
        # Validate classes using CLIP
        objects_with_validation = validate_class(
            image_path=image_path,
            detections=objects,
            anchors_path=anchors_path,
            similarity_threshold=similarity_threshold
        )
        
        # Log validation results
        invalid_objects = [obj for obj in objects_with_validation if not obj['validation']['valid']]
        if invalid_objects:
            logger.info(f"Found {len(invalid_objects)} invalid objects in {image_path}")
        
        # Apply OCR check
        objects_with_ocr = ocr_check(
            image_path=image_path,
            detections=objects_with_validation,
            regex_path=regex_path
        )
        
        # Log OCR results
        text_objects = [obj for obj in objects_with_ocr if obj.get('ocr', {}).get('has_text', False)]
        if text_objects:
            logger.info(f"Found {len(text_objects)} objects with text in {image_path}")
        
        # Create output paths
        output_image_path = output_img_dir / image_path.name
        output_label_path = Path(output_label_dir) / label_path.name
        
        # Write validated objects back to YOLO format
        with open(output_label_path, 'w') as f:
            for obj in objects_with_ocr:
                # Only include valid objects
                if obj.get('validation', {}).get('valid', True):
                    # Extract bbox coordinates
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Convert back to YOLO format (normalized)
                    x_center = (x1 + x2) / (2.0 * img_width)
                    y_center = (y1 + y2) / (2.0 * img_height)
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write YOLO format line
                    f.write(f"{obj['class_id']} {x_center} {y_center} {width} {height}\n")
        
        # Copy image to output directory
        shutil.copy2(image_path, output_image_path)
        
        return True
        
    except Exception as e:
        logger.exception(f"Error validating {image_path}: {e}")
        return False

def main(args):
    start_time = time.time()
    
    # Find all images in the input directory
    logger.info(f"Finding images in {args.input_dir}...")
    images = find_images(args.input_dir)
    logger.info(f"Found {len(images)} images to validate")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(tqdm(images, desc="Validating images")):
        # Process image
        result = validate_image(
            image_path=image_path,
            output_dir=args.output_dir,
            anchors_path=args.anchors_path,
            regex_path=args.regex_path,
            similarity_threshold=args.similarity
        )
        
        if result:
            successful += 1
        else:
            failed += 1
            
        # Log progress
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            elapsed = time.time() - start_time
            images_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(images) - (i + 1)) / images_per_sec if images_per_sec > 0 else 0
            
            logger.info(f"Progress: {i+1}/{len(images)} images ({(i+1)/len(images)*100:.1f}%)")
            logger.info(f"Time elapsed: {elapsed:.1f}s, {images_per_sec:.2f} img/s")
            logger.info(f"Estimated time remaining: {remaining:.1f}s")
            logger.info(f"Successful: {successful}, Failed: {failed}")
    
    # Final report
    logger.info(f"Validation complete!")
    logger.info(f"Total images: {len(images)}")
    logger.info(f"Successfully validated: {successful}")
    logger.info(f"Failed validation: {failed}")
    logger.info(f"Total processing time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply validation to cleaned dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with cleaned dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for validated dataset")
    parser.add_argument("--anchors_path", type=str, required=True, help="Path to anchors file for CLIP validation")
    parser.add_argument("--regex_path", type=str, required=True, help="Path to regex patterns for OCR check")
    parser.add_argument("--similarity", type=float, default=0.65, help="Similarity threshold for CLIP validation")
    
    args = parser.parse_args()
    main(args)
