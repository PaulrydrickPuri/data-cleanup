#!/usr/bin/env python3
"""
Modified version of clean_dataset.py that supports processing a single image.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from data_cleanup.detect_objects import detect_objects
from data_cleanup.segment_masks import segment_masks
from data_cleanup.validate_class import validate_class, ClassValidator
from data_cleanup.ocr_check import ocr_check, OCRChecker
from data_cleanup.quality_filter import quality_filter

# Persistent model instances - loaded once and reused
_CLIP_VALIDATOR = None
_OCR_CHECKER = None

def get_clip_validator(anchors_path, similarity_threshold=0.65):
    """Get or create a persistent ClassValidator instance."""
    global _CLIP_VALIDATOR
    if _CLIP_VALIDATOR is None:
        logger.info(f"Creating persistent ClassValidator with threshold {similarity_threshold}")
        _CLIP_VALIDATOR = ClassValidator(anchors_path, similarity_threshold)
    return _CLIP_VALIDATOR

def get_ocr_checker(regex_path):
    """Get or create a persistent OCRChecker instance."""
    global _OCR_CHECKER
    if _OCR_CHECKER is None:
        logger.info("Creating persistent OCRChecker")
        _OCR_CHECKER = OCRChecker(regex_path)
    return _OCR_CHECKER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_cleanup_single.log")
    ]
)
logger = logging.getLogger(__name__)

def clean_single_image(image_path: Union[str, Path],
                      output_dir: Union[str, Path],
                      anchors_path: Union[str, Path],
                      regex_path: Union[str, Path],
                      confidence_threshold: float = 0.25,
                      similarity_threshold: float = 0.65,
                      blur_threshold: float = 100.0,
                      min_size: int = 64,
                      min_exposure: int = 30,
                      max_exposure: int = 225,
                      log_raw_crops: bool = False) -> Dict:
    """
    Process a single image through the entire cleanup pipeline.
    
    Args:
        image_path: Path to the image file.
        output_dir: Directory to save the cleaned output.
        anchors_path: Path to JSON file with ground truth anchor embeddings.
        regex_path: Path to JSON file with regex patterns.
        confidence_threshold: Minimum confidence score for object detection.
        similarity_threshold: Minimum cosine similarity for class validation.
        blur_threshold: Minimum Laplacian variance to consider an image not blurry.
        min_size: Minimum width or height in pixels.
        min_exposure: Minimum mean pixel intensity.
        max_exposure: Maximum mean pixel intensity.
        log_raw_crops: Whether to save rejected crops for inspection.
        
    Returns:
        Statistics about the cleaning process.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Check if output_dir already has images/labels structure
    output_dir_str = str(output_dir)
    if 'images' in output_dir_str.split(os.sep):
        # Already contains an 'images' directory in the path
        images_dir = output_dir
        # Replace 'images' with 'labels' in the path
        labels_dir_parts = output_dir_str.split(os.sep)
        if 'images' in labels_dir_parts:
            idx = labels_dir_parts.index('images')
            labels_dir_parts[idx] = 'labels'
            labels_dir = Path(os.sep.join(labels_dir_parts))
        else:
            # Fallback: use parent dir and add 'labels'
            labels_dir = output_dir.parent / 'labels'
    else:
        # Create standard output directories
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
    
    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    logger.info(f"Using images_dir: {images_dir}")
    logger.info(f"Using labels_dir: {labels_dir}")
    
    # Initialize statistics
    stats = {
        'total_images': 1,
        'processed_images': 0,
        'rejected_images': 0,
        'total_objects': 0,
        'accepted_objects': 0,
        'rejected_objects': 0,
        'saved_images': 0,
        'saved_objects': 0,
        'rejection_reasons': {}
    }
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['load_failure'] = stats['rejection_reasons'].get('load_failure', 0) + 1
            return stats
            
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        # Keep original dimensions as integers for array indexing
        img_height_float, img_width_float = float(img_height), float(img_width)
        
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
            stats['rejected_images'] += 1
            stats['rejection_reasons']['no_label'] = stats['rejection_reasons'].get('no_label', 0) + 1
            return stats
            
        # Parse YOLO label
        objects = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to absolute coordinates
                    x1 = int((x_center - width/2.0) * img_width_float)
                    y1 = int((y_center - height/2.0) * img_height_float)
                    x2 = int((x_center + width/2.0) * img_width_float)
                    y2 = int((y_center + height/2.0) * img_height_float)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    objects.append({
                        'class_id': int(class_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': 1.0  # Original annotations are treated as 100% confident
                    })
        
        stats['total_objects'] += len(objects)
        
        if not objects:
            logger.warning(f"No objects found in label file: {label_path}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['no_objects'] = stats['rejection_reasons'].get('no_objects', 0) + 1
            return stats
            
        # Segment masks for objects
        try:
            # Ensure numeric types in objects list
            for obj in objects:
                obj['bbox'] = [float(x) for x in obj['bbox']]

            # Segment masks - SAM processes bounding boxes to create precise masks
            # This filters out boxes with too much background
            logger.info(f"Segmenting {len(objects)} objects for {image_path}")
            objects_with_masks = segment_masks(
                image_path=image_path,
                detections=objects,
                output_dir=output_dir / 'masks',
                sam_checkpoint=None  # Will use environment variable
            )
            
            # After SAM processing, we need to ensure objects have the correct format
            # for subsequent validation steps
            for obj in objects_with_masks:
                # Ensure bbox is in the correct format [x1, y1, x2, y2] with numeric values
                if 'bbox' in obj:
                    obj['bbox'] = [float(coord) for coord in obj['bbox']]
                # Make sure there's a class_id
                if 'class_id' not in obj and 'class' in obj:
                    try:
                        obj['class_id'] = int(obj['class'])
                    except (ValueError, TypeError):
                        obj['class_id'] = 0  # Default class ID if conversion fails
            
            logger.info(f"SAM processing complete: {len(objects_with_masks)} objects retained after filtering")
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['segmentation_error'] = stats['rejection_reasons'].get('segmentation_error', 0) + 1
            return stats
            
        # Implement selective validation - only validate on every 10th image to improve performance
        # Get image index from path to determine if we should validate
        image_index = hash(str(image_path)) % 1000  # Use hash to get consistent index
        should_validate = image_index % 10 == 0  # Validate every 10th image

        # Define class mapping from class_id to class name
        class_mapping = {
            0: "car",
            1: "truck",
            2: "bus",
            3: "motorcycle",
            4: "bicycle"
            # Add more classes as needed
        }

        try:
            # Step 3: CLIP Validation - process objects that passed SAM filtering
            if should_validate:
                logger.info(f"Performing class validation for {image_path} (image {image_index})")
                
                # Get persistent CLIP validator
                validator = get_clip_validator(anchors_path, similarity_threshold)
                
                # Load image once for all objects
                full_image = cv2.imread(str(image_path))
                if full_image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                objects_with_class = []
                for obj in objects_with_masks:
                    # Make sure class_id is present and properly mapped to class name
                    if 'class_id' in obj:
                        class_id = obj['class_id']
                        # Map class_id to proper class name
                        if class_id in class_mapping:
                            obj['class'] = class_mapping[class_id]
                        else:
                            obj['class'] = f"vehicle_{class_id}"
                    elif 'class' not in obj:
                        # Fallback if neither class nor class_id is present
                        obj['class'] = "vehicle"
                        obj['class_id'] = 0
                    
                    # Create a copy to avoid modifying the original
                    detection_copy = obj.copy()
                    
                    try:
                        # Ensure bbox is in the correct format [x1, y1, x2, y2] with integer values
                        if 'bbox' in obj:
                            # Extract object crop for validation (rectangular bounding box)
                            x1, y1, x2, y2 = [int(float(coord)) for coord in obj['bbox']]  # Convert to integers
                            
                            # Ensure coordinates are valid
                            h, w = full_image.shape[:2]
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(x1+1, min(x2, w))
                            y2 = max(y1+1, min(y2, h))
                            
                            # Extract the crop
                            crop = full_image[y1:y2, x1:x2]
                            
                            # Validate against CLIP embeddings
                            logger.debug(f"Validating object class '{obj['class']}' with bbox {[x1, y1, x2, y2]}")
                            validation_result = validator.validate(
                                image=crop,  # Pass the crop directly
                                bbox=[0, 0, crop.shape[1], crop.shape[0]],  # Use full crop coordinates
                                class_name=obj['class']
                            )
                            
                            # Add validation result to object
                            detection_copy['validation'] = validation_result
                        else:
                            # Missing bbox - add dummy validation
                            logger.warning(f"Object missing bbox, using dummy validation")
                            detection_copy['validation'] = {
                                'valid': True,
                                'top_class': detection_copy.get('class', 'vehicle'),
                                'similarity': 1.0,
                                'class_match': True
                            }
                        
                        objects_with_class.append(detection_copy)
                    except Exception as e:
                        logger.warning(f"Error validating object: {str(e)}, using original object")
                        # Add dummy validation results for failed objects
                        detection_copy['validation'] = {
                            'valid': True,
                            'top_class': detection_copy.get('class', 'vehicle'),
                            'similarity': 1.0,
                            'class_match': True
                        }
                        objects_with_class.append(detection_copy)
                
                # Count invalid classifications
                invalid_objects = [obj for obj in objects_with_class if not obj.get('validation', {}).get('valid', True)]
                if invalid_objects:
                    logger.info(f"Found {len(invalid_objects)} invalid objects in {image_path}")
                    stats['rejected_objects'] += len(invalid_objects)
            else:
                # Skip validation for this image to improve performance
                logger.info(f"Skipping validation for {image_path} to improve performance")
                objects_with_class = objects_with_masks.copy()
                
                # Add dummy validation results
                for obj in objects_with_class:
                    # Ensure class mapping is applied for consistency
                    if 'class_id' in obj and 'class' not in obj:
                        class_id = obj['class_id']
                        obj['class'] = class_mapping.get(class_id, f"vehicle_{class_id}")
                    elif 'class' not in obj:
                        obj['class'] = "vehicle"  # Default class
                        
                    # Add dummy validation result
                    obj['validation'] = {
                        'valid': True,
                        'top_class': obj.get('class', 'vehicle'),
                        'similarity': 1.0,
                        'class_match': True
                    }
                
        except Exception as e:
            logger.error(f"Error during class validation: {e}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['validation_error'] = stats['rejection_reasons'].get('validation_error', 0) + 1
            return stats
            
        # Step 4: OCR Check - detect license plates and text in vehicle images
        # Use the same image_index to determine whether to run OCR check
        # This ensures consistency with validation (same images get validated and OCR checked)
        try:
            if should_validate:  # Reuse the same selection as for validation
                logger.info(f"Performing OCR check for {image_path} (image {image_index})")
                
                # Get persistent OCR checker
                checker = get_ocr_checker(regex_path)
                
                # Load image once for all objects (if not already loaded)
                if 'full_image' not in locals() or full_image is None:
                    full_image = cv2.imread(str(image_path))
                    if full_image is None:
                        raise ValueError(f"Failed to load image: {image_path}")
                
                objects_with_ocr = []
                for obj in objects_with_class:
                    try:
                        # Create a copy to avoid modifying the original
                        detection_copy = obj.copy()
                        
                        # Only process potential license plate objects
                        plate_related_classes = ['carplate', 'license_plate', 'plate']
                        if any(plate_class in str(detection_copy.get('class', '')).lower() for plate_class in plate_related_classes):
                            logger.debug(f"Processing potential license plate object: {detection_copy.get('class')}")
                            
                            # Ensure bbox is in the correct format with integer values
                            if 'bbox' in obj:
                                # Extract license plate crop for OCR
                                try:
                                    x1, y1, x2, y2 = [int(float(coord)) for coord in obj['bbox']]  # Convert to integers
                                    
                                    # Ensure coordinates are valid
                                    h, w = full_image.shape[:2]
                                    x1 = max(0, min(x1, w-1))
                                    y1 = max(0, min(y1, h-1))
                                    x2 = max(x1+1, min(x2, w))
                                    y2 = max(y1+1, min(y2, h))
                                    
                                    # Extract the crop
                                    crop = full_image[y1:y2, x1:x2]
                                    
                                    # Check plate directly using the loaded checker
                                    ocr_result = checker.check_plate(
                                        image=crop,  # Pass the crop directly
                                        bbox=[0, 0, crop.shape[1], crop.shape[0]],  # Full crop coordinates
                                        preprocess=True
                                    )
                                    
                                    # Add OCR result to object with privacy protection
                                    detection_copy['ocr'] = {
                                        'has_text': bool(ocr_result.get('text', '')),
                                        'text': '',  # Don't store actual text for privacy
                                        'valid': ocr_result.get('verified', True)
                                    }
                                except Exception as e:
                                    logger.warning(f"Error processing license plate crop: {e}")
                                    detection_copy['ocr'] = {
                                        'has_text': False,
                                        'text': '',
                                        'valid': True,
                                        'error': str(e)
                                    }
                            else:
                                # Missing bbox - add default OCR results
                                detection_copy['ocr'] = {
                                    'has_text': False,
                                    'text': '',
                                    'valid': True,
                                    'note': 'missing_bbox'
                                }
                        else:
                            # Non-plate objects get default OCR results
                            detection_copy['ocr'] = {
                                'has_text': False,
                                'text': '',
                                'valid': True
                            }
                            
                        objects_with_ocr.append(detection_copy)
                    except Exception as e:
                        logger.warning(f"Error in OCR check for object: {str(e)}, using original object")
                        # Add default OCR results for compatibility
                        obj['ocr'] = {
                            'has_text': False,
                            'text': '',
                            'valid': True,
                            'error': str(e)
                        }
                        objects_with_ocr.append(obj)
                        
                # Log objects with text (potential license plates)
                text_objects = [obj for obj in objects_with_ocr if obj.get('ocr', {}).get('has_text', False)]
                if text_objects:
                    logger.info(f"Found {len(text_objects)} objects with text in {image_path}")
            else:
                # Skip OCR for this image to improve performance
                logger.info(f"Skipping OCR check for {image_path} to improve performance")
                objects_with_ocr = objects_with_class.copy()
                
                # Add dummy OCR results to maintain consistency
                for obj in objects_with_ocr:
                    obj['ocr'] = {
                        'has_text': False,
                        'text': '',
                        'valid': True
                    }
            
            # Log objects with text (potential license plates)
            text_objects = [obj for obj in objects_with_ocr if obj.get('ocr', {}).get('has_text', False)]
            if text_objects:
                logger.info(f"Found {len(text_objects)} objects with text in {image_path}")
                
        except Exception as e:
            logger.error(f"Error during OCR check: {e}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['ocr_error'] = stats['rejection_reasons'].get('ocr_error', 0) + 1
            return stats
            
        # Step 5: Quality Filtering - final validation before dataset output
        try:
            logger.info(f"Applying quality filtering to {len(objects_with_ocr)} objects in {image_path}")
            
            # Apply quality filtering to objects that passed previous stages
            objects_filtered = quality_filter(
                image_path=image_path,
                detections=objects_with_ocr,
                blur_threshold=blur_threshold,
                min_size=min_size,
                min_exposure=min_exposure,
                max_exposure=max_exposure
            )
            
            # Count rejected objects
            rejected_objects = len(objects_with_ocr) - len(objects_filtered)
            stats['rejected_objects'] += rejected_objects
            
            # Log quality filtering results
            if rejected_objects > 0:
                logger.info(f"Rejected {rejected_objects} objects due to quality issues (blur, size, exposure)")
                
                # Log specific rejection reasons if available
                blur_rejected = sum(1 for obj in objects_with_ocr if obj.get('quality', {}).get('reject_reason') == 'blur')
                size_rejected = sum(1 for obj in objects_with_ocr if obj.get('quality', {}).get('reject_reason') == 'size')
                exposure_rejected = sum(1 for obj in objects_with_ocr if obj.get('quality', {}).get('reject_reason') == 'exposure')
                
                if blur_rejected > 0:
                    logger.info(f"  - {blur_rejected} objects rejected due to blur")
                if size_rejected > 0:
                    logger.info(f"  - {size_rejected} objects rejected due to size")
                if exposure_rejected > 0:
                    logger.info(f"  - {exposure_rejected} objects rejected due to exposure")
                
            # If all objects were rejected, reject the entire image
            if len(objects_filtered) == 0 and len(objects_with_ocr) > 0:
                logger.info(f"All objects in {image_path} were rejected due to quality issues")
                stats['rejected_images'] += 1
                stats['rejection_reasons']['quality_filter'] = stats['rejection_reasons'].get('quality_filter', 0) + 1
                return stats
                
        except Exception as e:
            logger.error(f"Error during quality filtering: {e}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['quality_filter_error'] = stats['rejection_reasons'].get('quality_filter_error', 0) + 1
            return stats
            
        # Step 6: Save Cleaned Image and Labels - final output
        try:
            # Use the dataset structure to create the output path
            # First get the top level dataset directory (e.g., 'Dataset')
            input_path = str(image_path)
            # Extract only the relevant structure for output (e.g., '1/Charge-EV/train/images')
            # This is a simpler approach that preserves structure without duplication
            if 'Dataset' in input_path:
                # Remove dataset part from path
                rel_path = input_path.split('Dataset')[1].lstrip('\\/')
                # But don't include 'Dataset' itself in the output path
                output_image_path = output_dir / rel_path
            else:
                # Fallback in case 'Dataset' is not in the path
                output_image_path = output_dir / Path(image_path).name
                
            output_image_dir = output_image_path.parent
            os.makedirs(output_image_dir, exist_ok=True)
            
            # Create output label path
            label_name = image_path.with_suffix('.txt').name
            output_label_dir = str(output_image_dir).replace('images', 'labels')
            os.makedirs(output_label_dir, exist_ok=True)
            output_label_path = Path(output_label_dir) / label_name
            
            # Log paths for debugging
            logger.info(f"image_path type: {type(image_path)}, value: {image_path}")
            logger.info(f"images_dir type: {type(images_dir)}, value: {images_dir}")
            logger.info(f"labels_dir type: {type(labels_dir)}, value: {labels_dir}")
            logger.info(f"output_image_path: {output_image_path}")
            logger.info(f"output_label_path: {output_label_path}")
            
            # Copy image to output directory
            shutil.copy2(image_path, output_image_path)
            
            # Write YOLO format labels for objects that passed all filters
            with open(output_label_path, 'w') as f:
                for obj in objects_filtered:  # Using objects_filtered from quality filter step
                    class_id = obj['class_id']
                    x1, y1, x2, y2 = [float(coord) for coord in obj['bbox']]  # Ensure float values
                    
                    # Convert to YOLO format (normalized)
                    x_center = (x1 + x2) / (2 * img_width_float)
                    y_center = (y1 + y2) / (2 * img_height_float)
                    width = (x2 - x1) / img_width_float
                    height = (y2 - y1) / img_height_float
                    
                    # Write YOLO format line
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
            # Update statistics
            stats['saved_images'] += 1
            stats['saved_objects'] += len(objects_filtered)
            logger.info(f"Successfully saved cleaned image with {len(objects_filtered)} objects")
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            stats['rejected_images'] += 1
            stats['rejection_reasons']['save_error'] = stats['rejection_reasons'].get('save_error', 0) + 1
            return stats

    except Exception:
        logger.exception(f"Error processing {image_path}")
        stats['rejected_images'] += 1
        stats['rejection_reasons']['processing_error'] = stats['rejection_reasons'].get('processing_error', 0) + 1
        
    return stats
