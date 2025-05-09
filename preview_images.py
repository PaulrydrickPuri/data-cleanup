#!/usr/bin/env python3
"""
Script to preview images with their annotations for ground truth selection.
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml

def load_classes(yaml_path):
    """Load class names from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def draw_annotations(image, annotations, img_width, img_height, class_names):
    """Draw bounding boxes and labels on image."""
    colors = {
        'vehicle': (0, 255, 0),      # Green
        'carplate': (255, 0, 0),     # Blue
        'motorcycle': (0, 0, 255),   # Red
        'logo': (255, 255, 0)        # Cyan
    }
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            cls_name = class_names[cls_id]
            color = colors.get(cls_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def preview_images(dataset_path, image_paths, output_dir):
    """Preview images with annotations and save them to output directory."""
    # Load class names
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = load_classes(yaml_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        img_path = Path(img_path)
        img_name = img_path.name
        
        # Find corresponding label file
        label_path = Path(dataset_path) / "train" / "labels" / img_path.with_suffix(".txt").name
        
        if not label_path.exists():
            print(f"Warning: No label file found for {img_name}")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Failed to load image {img_name}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        # Draw annotations
        annotated_image = draw_annotations(image.copy(), annotations, img_width, img_height, class_names)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"preview_{img_name}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved preview for {img_name}")

def main():
    parser = argparse.ArgumentParser(description="Preview images with annotations for ground truth selection")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to preview")
    parser.add_argument("--output_dir", default="temp_preview", help="Directory to save previews")
    
    args = parser.parse_args()
    
    # Find image paths
    image_dir = os.path.join(args.dataset_path, "train", "images")
    image_paths = list(Path(image_dir).glob("*.jpg"))[:args.num_images]
    
    preview_images(args.dataset_path, image_paths, args.output_dir)

if __name__ == "__main__":
    main()
