#!/usr/bin/env python3
"""
Convert cleaned dataset to COCO JSON format.
"""
import os
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CocoConverter:
    """Convert YOLO format dataset to COCO JSON format."""
    
    def __init__(self, 
                 images_dir: Union[str, Path],
                 labels_dir: Union[str, Path],
                 masks_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the COCO converter.
        
        Args:
            images_dir: Directory containing the images.
            labels_dir: Directory containing the YOLO format labels.
            masks_dir: Optional directory containing the masks.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        
        # Initialize COCO dataset structure
        self.coco_dataset = {
            "info": {
                "description": "Vehicle Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CoreframeAI",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/4.0/"
                }
            ],
            "categories": [
                {"id": 1, "name": "vehicle", "supercategory": "vehicle"},
                {"id": 2, "name": "carplate", "supercategory": "vehicle"},
                {"id": 3, "name": "logo", "supercategory": "vehicle"}
            ],
            "images": [],
            "annotations": []
        }
        
        # Class name to ID mapping
        self.class_map = {
            "vehicle": 1,
            "carplate": 2,
            "logo": 3
        }
        
        # Keep track of annotation ID
        self.annotation_id = 1
        
    def _find_images(self) -> List[Path]:
        """
        Find all images in the images directory.
        
        Returns:
            List of paths to image files.
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(self.images_dir.glob(f'*{ext}')))
            
        return image_paths
        
    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """
        Find the corresponding label file for an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Path to the label file, or None if not found.
        """
        label_path = self.labels_dir / image_path.with_suffix('.txt').name
        
        if label_path.exists():
            return label_path
        else:
            return None
            
    def _find_mask_file(self, image_path: Path, obj_idx: int) -> Optional[Path]:
        """
        Find the corresponding mask file for an object.
        
        Args:
            image_path: Path to the image file.
            obj_idx: Index of the object.
            
        Returns:
            Path to the mask file, or None if not found.
        """
        if not self.masks_dir:
            return None
            
        mask_pattern = f"{image_path.stem}_mask_{obj_idx:03d}.png"
        mask_path = self.masks_dir / mask_pattern
        
        if mask_path.exists():
            return mask_path
        else:
            return None
            
    def _parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse a YOLO format label file.
        
        Args:
            label_path: Path to the label file.
            img_width: Width of the image.
            img_height: Height of the image.
            
        Returns:
            List of dictionaries with keys 'category_id', 'bbox', etc.
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
                    
                    # Map class ID to name and COCO category ID
                    class_map = {0: 'vehicle', 1: 'carplate', 2: 'logo'}
                    cls_name = class_map.get(cls_id, f'class_{cls_id}')
                    category_id = self.class_map.get(cls_name, 1)  # Default to vehicle if unknown
                    
                    objects.append({
                        'category_id': category_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                        'class_name': cls_name,
                        'yolo': {
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        }
                    })
                    
        return objects
        
    def _encode_mask_rle(self, mask_path: Path) -> Dict:
        """
        Encode a binary mask as RLE (Run-Length Encoding).
        
        Args:
            mask_path: Path to the mask file.
            
        Returns:
            Dictionary with RLE encoding.
        """
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            logger.warning(f"Failed to load mask: {mask_path}")
            return None
            
        # Binarize if not already binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
            
        # Compute RLE
        counts = []
        last_val = 0
        count = 0
        
        # Flatten mask
        flat_mask = mask.flatten()
        
        for pixel in flat_mask:
            if pixel == last_val:
                count += 1
            else:
                counts.append(count)
                last_val = pixel
                count = 1
                
        # Add final count
        counts.append(count)
        
        # COCO RLE format
        rle = {
            'counts': counts,
            'size': list(mask.shape)
        }
        
        return rle
        
    def convert(self, output_path: Union[str, Path]) -> None:
        """
        Convert the dataset to COCO JSON format.
        
        Args:
            output_path: Path to save the COCO JSON file.
        """
        # Find all images
        image_paths = self._find_images()
        
        logger.info(f"Found {len(image_paths)} images in {self.images_dir}")
        
        # Process each image
        for image_idx, image_path in enumerate(tqdm(image_paths, desc="Converting to COCO")):
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Add image to COCO dataset
            image_id = image_idx + 1
            self.coco_dataset['images'].append({
                'id': image_id,
                'file_name': image_path.name,
                'width': img_width,
                'height': img_height,
                'license': 1,
                'date_captured': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Find label file
            label_path = self._find_label_file(image_path)
            
            if not label_path:
                logger.warning(f"No label file found for {image_path}")
                continue
                
            # Parse label file
            objects = self._parse_yolo_label(label_path, img_width, img_height)
            
            # Add annotations to COCO dataset
            for obj_idx, obj in enumerate(objects):
                annotation = {
                    'id': self.annotation_id,
                    'image_id': image_id,
                    'category_id': obj['category_id'],
                    'bbox': obj['bbox'],
                    'area': obj['bbox'][2] * obj['bbox'][3],
                    'iscrowd': 0
                }
                
                # Add segmentation if mask available
                mask_path = self._find_mask_file(image_path, obj_idx)
                if mask_path:
                    rle = self._encode_mask_rle(mask_path)
                    if rle:
                        annotation['segmentation'] = rle
                        
                self.coco_dataset['annotations'].append(annotation)
                self.annotation_id += 1
                
        # Save COCO dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)
            
        logger.info(f"Saved COCO dataset with {len(self.coco_dataset['images'])} images and {len(self.coco_dataset['annotations'])} annotations to {output_path}")


def convert_to_coco(images_dir: Union[str, Path],
                   labels_dir: Union[str, Path],
                   output_path: Union[str, Path],
                   masks_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Convenience function to convert a dataset to COCO JSON format.
    
    Args:
        images_dir: Directory containing the images.
        labels_dir: Directory containing the YOLO format labels.
        output_path: Path to save the COCO JSON file.
        masks_dir: Optional directory containing the masks.
    """
    converter = CocoConverter(images_dir, labels_dir, masks_dir)
    converter.convert(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLO format dataset to COCO JSON format")
    parser.add_argument("--images", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--labels", type=str, required=True, help="Directory containing the YOLO format labels")
    parser.add_argument("--masks", type=str, default=None, help="Optional directory containing the masks")
    parser.add_argument("--output", type=str, default="instances_train.json", help="Path to save the COCO JSON file")
    
    args = parser.parse_args()
    
    try:
        convert_to_coco(
            args.images, 
            args.labels, 
            args.output, 
            args.masks
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
