#!/usr/bin/env python3
"""
Pipeline orchestrator for cleaning and filtering a vehicle detection dataset.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from dotenv import load_dotenv
import cv2
import numpy as np
from tqdm import tqdm

from data_cleanup.detect_objects import detect_objects
from data_cleanup.segment_masks import segment_masks
from data_cleanup.validate_class import validate_class
from data_cleanup.ocr_check import ocr_check
from data_cleanup.quality_filter import quality_filter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_cleanup.log")
    ]
)
logger = logging.getLogger(__name__)

# JSON handler for structured logging
class JsonFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        
    def emit(self, record):
        try:
            if hasattr(record, 'json_data'):
                # Write JSON data directly
                self.stream.write(json.dumps(record.json_data))
                self.stream.write('\n')
                self.flush()
        except Exception:
            self.handleError(record)

# Add JSON handler if enabled
if os.getenv('LOG_JSON', 'true').lower() == 'true':
    json_handler = JsonFileHandler('data_cleanup_audit.json')
    logger.addHandler(json_handler)

class DatasetCleaner:
    """Pipeline orchestrator for cleaning and filtering a vehicle detection dataset."""
    
    def __init__(self, 
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 anchors_path: Union[str, Path],
                 regex_path: Union[str, Path],
                 confidence_threshold: float = 0.25,
                 similarity_threshold: float = 0.65,
                 blur_threshold: float = 100.0,
                 min_size: int = 64,
                 min_exposure: int = 30,
                 max_exposure: int = 225,
                 log_raw_crops: bool = False):
        """
        Initialize the dataset cleaner.
        
        Args:
            input_dir: Directory containing the input dataset.
            output_dir: Directory to save the cleaned dataset.
            anchors_path: Path to JSON file with ground truth anchor embeddings.
            regex_path: Path to JSON file with regex patterns.
            confidence_threshold: Minimum confidence score for object detection.
            similarity_threshold: Minimum cosine similarity for class validation.
            blur_threshold: Minimum Laplacian variance to consider an image not blurry.
            min_size: Minimum width or height in pixels.
            min_exposure: Minimum mean pixel intensity.
            max_exposure: Maximum mean pixel intensity.
            log_raw_crops: Whether to save rejected crops for inspection.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.anchors_path = Path(anchors_path)
        self.regex_path = Path(regex_path)
        
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.blur_threshold = blur_threshold
        self.min_size = min_size
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        
        # Override log_raw_crops from environment variable if present
        log_raw_crops_env = os.getenv('LOG_RAW_CROPS', str(log_raw_crops).lower())
        self.log_raw_crops = log_raw_crops_env.lower() == 'true'
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.masks_dir = self.output_dir / 'masks'
        self.rejected_dir = self.output_dir / 'rejected' if self.log_raw_crops else None
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        if self.rejected_dir:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'rejected_images': 0,
            'total_objects': 0,
            'accepted_objects': 0,
            'rejected_objects': 0,
            'rejection_reasons': {}
        }
        
    def _find_images(self) -> List[Path]:
        """
        Find all images in the input directory.
        
        Returns:
            List of paths to image files.
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(self.input_dir.glob(f'**/*{ext}')))
            
        return image_paths
        
    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """
        Find the corresponding label file for an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Path to the label file, or None if not found.
        """
        # Try different label extensions
        label_extensions = ['.txt', '.xml', '.json']
        
        for ext in label_extensions:
            label_path = image_path.with_suffix(ext)
            if label_path.exists():
                return label_path
                
        # Try YOLO format (same name but in 'labels' directory)
        rel_path = image_path.relative_to(self.input_dir)
        if 'images' in rel_path.parts:
            # Replace 'images' with 'labels' in the path
            parts = list(rel_path.parts)
            idx = parts.index('images')
            parts[idx] = 'labels'
            label_path = self.input_dir.joinpath(*parts).with_suffix('.txt')
            if label_path.exists():
                return label_path
                
        return None
        
    def _parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse a YOLO format label file.
        
        Args:
            label_path: Path to the label file.
            img_width: Width of the image.
            img_height: Height of the image.
            
        Returns:
            List of dictionaries with keys 'class', 'bbox', etc.
        """
        objects = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to absolute coordinates
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    # Map class ID to name
                    class_map = {0: 'vehicle', 1: 'carplate', 2: 'logo'}
                    cls_name = class_map.get(cls_id, f'class_{cls_id}')
                    
                    objects.append({
                        'class': cls_name,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': cls_id,
                        'yolo': {
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        }
                    })
                    
        return objects
        
    def _write_yolo_label(self, objects: List[Dict], label_path: Path, img_width: int, img_height: int) -> None:
        """
        Write objects to a YOLO format label file.
        
        Args:
            objects: List of dictionaries with keys 'class_id', 'bbox', etc.
            label_path: Path to save the label file.
            img_width: Width of the image.
            img_height: Height of the image.
        """
        with open(label_path, 'w') as f:
            for obj in objects:
                cls_id = obj['class_id']
                x1, y1, x2, y2 = obj['bbox']
                
                # Convert to YOLO format
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                
    def _log_rejection(self, image_path: Path, objects: List[Dict], reason: str) -> None:
        """
        Log a rejection for audit purposes.
        
        Args:
            image_path: Path to the image file.
            objects: List of rejected objects.
            reason: Reason for rejection.
        """
        # Update statistics
        if reason not in self.stats['rejection_reasons']:
            self.stats['rejection_reasons'][reason] = 0
        self.stats['rejection_reasons'][reason] += 1
        
        # Prepare audit data
        audit_data = {
            'image_path': str(image_path),
            'reason': reason,
            'timestamp': logging.Formatter.formatTime(logging.Formatter(), logging.LogRecord('', 0, '', 0, None, None, None)),
            'objects': []
        }
        
        # Add object data (without actual crops)
        for obj in objects:
            obj_data = {
                'class': obj['class'],
                'bbox': obj['bbox']
            }
            
            if 'validation' in obj:
                obj_data['validation'] = {
                    'valid': obj['validation']['valid'],
                    'top_class': obj['validation']['top_class'],
                    'similarity': obj['validation']['similarity']
                }
                
            if 'quality' in obj:
                obj_data['quality'] = {
                    'is_good_quality': obj['quality']['is_good_quality'],
                    'reason': obj['quality']['reason']
                }
                
            audit_data['objects'].append(obj_data)
            
        # Log as JSON
        log_record = logging.LogRecord('', logging.INFO, '', 0, '', (), None)
        log_record.json_data = audit_data
        for handler in logger.handlers:
            if isinstance(handler, JsonFileHandler):
                handler.emit(log_record)
                
        # Save rejected crops if enabled
        if self.log_raw_crops and self.rejected_dir:
            # Load image
            image = cv2.imread(str(image_path))
            if image is not None:
                # Save full image
                rejected_img_path = self.rejected_dir / f"{image_path.stem}_{reason.replace(' ', '_')}{image_path.suffix}"
                cv2.imwrite(str(rejected_img_path), image)
                
                # Save crops
                for i, obj in enumerate(objects):
                    x1, y1, x2, y2 = obj['bbox']
                    crop = image[y1:y2, x1:x2]
                    crop_path = self.rejected_dir / f"{image_path.stem}_{i}_{obj['class']}_{reason.replace(' ', '_')}{image_path.suffix}"
                    cv2.imwrite(str(crop_path), crop)
                
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image through the entire pipeline.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Whether the image was successfully processed and kept.
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                self._log_rejection(image_path, [], "failed_to_load")
                return False
                
            img_height, img_width = image.shape[:2]
            
            # Find label file
            label_path = self._find_label_file(image_path)
            
            # Get objects from label file or detect them
            if label_path:
                objects = self._parse_yolo_label(label_path, img_width, img_height)
                logger.info(f"Loaded {len(objects)} objects from {label_path}")
            else:
                # Detect objects
                objects = detect_objects(
                    image_path, 
                    confidence_threshold=self.confidence_threshold
                )
                logger.info(f"Detected {len(objects)} objects in {image_path}")
                
            if not objects:
                logger.warning(f"No objects found in {image_path}")
                self._log_rejection(image_path, [], "no_objects")
                return False
                
            # Update statistics
            self.stats['total_objects'] += len(objects)
            
            # Generate masks
            objects_with_masks = segment_masks(
                image_path, 
                objects, 
                self.masks_dir
            )
            
            # Validate classes
            validated_objects = []
            for obj in objects_with_masks:
                # Validate class
                validated_obj = validate_class(
                    image_path, 
                    obj, 
                    self.anchors_path, 
                    self.similarity_threshold
                )
                
                # Check if valid
                if not validated_obj['validation']['valid']:
                    logger.info(f"Rejected {obj['class']} due to invalid class validation")
                    self._log_rejection(image_path, [validated_obj], "invalid_class")
                    continue
                    
                validated_objects.append(validated_obj)
                
            if not validated_objects:
                logger.warning(f"No objects passed class validation in {image_path}")
                self._log_rejection(image_path, objects_with_masks, "no_valid_classes")
                return False
                
            # OCR check for carplates
            ocr_objects = []
            for obj in validated_objects:
                if obj['class'] == 'carplate':
                    # OCR check
                    ocr_obj = ocr_check(
                        image_path, 
                        obj, 
                        self.regex_path
                    )
                    
                    # Check if verified
                    if not ocr_obj['ocr_verified']:
                        logger.info(f"Rejected carplate due to invalid OCR")
                        self._log_rejection(image_path, [ocr_obj], "invalid_ocr")
                        continue
                        
                    ocr_objects.append(ocr_obj)
                else:
                    ocr_objects.append(obj)
                    
            # Quality check
            quality_objects = quality_filter(
                image_path, 
                ocr_objects, 
                self.blur_threshold, 
                self.min_size, 
                self.min_exposure, 
                self.max_exposure
            )
            
            # Filter out low quality objects
            good_quality_objects = []
            for obj in quality_objects:
                if not obj['is_good_quality']:
                    logger.info(f"Rejected {obj['class']} due to low quality: {obj['quality']['reason']}")
                    self._log_rejection(image_path, [obj], f"low_quality_{obj['quality']['reason']}")
                    continue
                    
                good_quality_objects.append(obj)
                
            if not good_quality_objects:
                logger.warning(f"No objects passed quality check in {image_path}")
                self._log_rejection(image_path, quality_objects, "no_good_quality_objects")
                return False
                
            # Update statistics
            self.stats['accepted_objects'] += len(good_quality_objects)
            self.stats['rejected_objects'] += len(objects) - len(good_quality_objects)
            
            # Save cleaned image and labels
            output_image_path = self.images_dir / image_path.name
            output_label_path = self.labels_dir / image_path.with_suffix('.txt').name
            
            # Copy image
            shutil.copy2(image_path, output_image_path)
            
            # Write YOLO label
            self._write_yolo_label(good_quality_objects, output_label_path, img_width, img_height)
            
            logger.info(f"Saved cleaned image to {output_image_path} with {len(good_quality_objects)} objects")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            self._log_rejection(image_path, [], f"error_{str(e)}")
            return False
            
    def clean_dataset(self) -> Dict:
        """
        Clean the entire dataset.
        
        Returns:
            Statistics about the cleaning process.
        """
        # Find all images
        image_paths = self._find_images()
        self.stats['total_images'] = len(image_paths)
        
        logger.info(f"Found {len(image_paths)} images in {self.input_dir}")
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Cleaning dataset"):
            success = self.process_image(image_path)
            
            if success:
                self.stats['processed_images'] += 1
            else:
                self.stats['rejected_images'] += 1
                
        # Log statistics
        logger.info(f"Cleaned dataset statistics: {self.stats}")
        
        # Save statistics
        stats_path = self.output_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        return self.stats


def clean_dataset(input_dir: Union[str, Path],
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
    Convenience function to clean a dataset.
    
    Args:
        input_dir: Directory containing the input dataset.
        output_dir: Directory to save the cleaned dataset.
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
    cleaner = DatasetCleaner(
        input_dir, 
        output_dir, 
        anchors_path, 
        regex_path, 
        confidence_threshold, 
        similarity_threshold, 
        blur_threshold, 
        min_size, 
        min_exposure, 
        max_exposure, 
        log_raw_crops
    )
    
    return cleaner.clean_dataset()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean and filter a vehicle detection dataset")
    parser.add_argument("input_dir", type=str, help="Directory containing the input dataset")
    parser.add_argument("output_dir", type=str, help="Directory to save the cleaned dataset")
    parser.add_argument("--anchors", type=str, default="assets/gt_anchors.json", help="Path to anchor embeddings JSON file")
    parser.add_argument("--regex", type=str, default="assets/regex_plate.json", help="Path to regex patterns JSON file")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for object detection")
    parser.add_argument("--similarity", type=float, default=0.65, help="Similarity threshold for class validation")
    parser.add_argument("--blur", type=float, default=100.0, help="Blur threshold")
    parser.add_argument("--min-size", type=int, default=64, help="Minimum size")
    parser.add_argument("--min-exposure", type=int, default=30, help="Minimum exposure")
    parser.add_argument("--max-exposure", type=int, default=225, help="Maximum exposure")
    parser.add_argument("--log-raw-crops", action="store_true", help="Save rejected crops for inspection")
    
    args = parser.parse_args()
    
    try:
        stats = clean_dataset(
            args.input_dir, 
            args.output_dir, 
            args.anchors, 
            args.regex, 
            args.confidence, 
            args.similarity, 
            args.blur, 
            args.min_size, 
            args.min_exposure, 
            args.max_exposure, 
            args.log_raw_crops
        )
        
        print(f"Cleaned dataset statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Processed images: {stats['processed_images']}")
        print(f"  Rejected images: {stats['rejected_images']}")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Accepted objects: {stats['accepted_objects']}")
        print(f"  Rejected objects: {stats['rejected_objects']}")
        print(f"  Rejection reasons:")
        for reason, count in stats['rejection_reasons'].items():
            print(f"    {reason}: {count}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
