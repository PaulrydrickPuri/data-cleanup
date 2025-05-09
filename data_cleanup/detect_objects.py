#!/usr/bin/env python3
"""
Object detection module using YOLOv8 for vehicle, carplate, and logo detection.
"""
import os
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """YOLOv8 detector for vehicle, carplate, and logo detection."""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLOv8 model weights. If None, uses YOLOv8n.
            confidence_threshold: Minimum confidence score to consider a detection valid.
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom YOLOv8 model from {model_path}")
        else:
            # Default to YOLOv8n if no model path provided
            self.model = YOLO("yolov8n.pt")
            logger.info("Loaded default YOLOv8n model")
            
        # Define class mapping for our specific classes
        self.class_map = {
            0: "vehicle",
            1: "carplate", 
            2: "logo"
        }
            
    def detect(self, image_path: Union[str, Path]) -> List[Dict]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            List of dictionaries with keys 'class', 'bbox', 'score'.
            bbox format is [x1, y1, x2, y2] in absolute pixel coordinates.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If no detections meet the confidence threshold.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Run inference
        results = self.model(image)
        
        # Process results
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Get confidence score
                conf = float(box.conf[0])
                
                # Skip if below threshold
                if conf < self.confidence_threshold:
                    continue
                    
                # Get class
                cls_id = int(box.cls[0])
                cls_name = self.class_map.get(cls_id, f"class_{cls_id}")
                
                # Get bounding box (convert to [x1, y1, x2, y2] format)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "score": conf
                })
        
        if not detections:
            logger.warning(f"No objects detected above threshold {self.confidence_threshold} in {image_path}")
            
        return detections


def detect_objects(image_path: Union[str, Path], 
                  model_path: Optional[str] = None,
                  confidence_threshold: float = 0.25) -> List[Dict]:
    """
    Convenience function to detect objects in an image.
    
    Args:
        image_path: Path to the image file.
        model_path: Path to YOLOv8 model weights. If None, uses YOLOv8n.
        confidence_threshold: Minimum confidence score to consider a detection valid.
        
    Returns:
        List of dictionaries with keys 'class', 'bbox', 'score'.
    """
    detector = ObjectDetector(model_path, confidence_threshold)
    return detector.detect(image_path)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Detect vehicles, carplates, and logos in images")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        results = detect_objects(args.image_path, args.model, args.conf)
        
        # Print results
        print(f"Detected {len(results)} objects:")
        for i, det in enumerate(results):
            print(f"{i+1}. {det['class']} (score: {det['score']:.2f}): {det['bbox']}")
            
        # Save to JSON if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
