#!/usr/bin/env python3
"""
Script to select and display ground truth images with their annotations.
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import shutil

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

def count_classes_in_annotation(annotation_path, num_classes):
    """Count the number of instances of each class in an annotation file."""
    if not os.path.exists(annotation_path):
        return [0] * num_classes
    
    class_counts = [0] * num_classes
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if 0 <= cls_id < num_classes:
                    class_counts[cls_id] += 1
    
    return class_counts

def select_gt_images(dataset_path, output_dir, num_images=10):
    """Select ground truth images with good class distribution."""
    # Load class names
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = load_classes(yaml_path)
    num_classes = len(class_names)
    
    # Find all image paths
    image_dir = os.path.join(dataset_path, "train", "images")
    label_dir = os.path.join(dataset_path, "train", "labels")
    
    image_paths = list(Path(image_dir).glob("*.jpg"))
    
    # Score images based on class distribution
    image_scores = []
    for img_path in image_paths:
        label_path = os.path.join(label_dir, img_path.with_suffix(".txt").name)
        class_counts = count_classes_in_annotation(label_path, num_classes)
        
        # Score based on number of classes present and total objects
        num_classes_present = sum(1 for count in class_counts if count > 0)
        total_objects = sum(class_counts)
        
        # Prioritize images with more classes and a reasonable number of objects
        score = num_classes_present * 10 + min(total_objects, 10)
        
        image_scores.append((img_path, score, class_counts))
    
    # Sort by score (descending)
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top images
    selected_images = image_scores[:num_images]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process selected images
    results = []
    for i, (img_path, score, class_counts) in enumerate(selected_images):
        img_name = img_path.name
        
        # Find corresponding label file
        label_path = os.path.join(label_dir, img_path.with_suffix(".txt").name)
        
        if not os.path.exists(label_path):
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
        output_path = os.path.join(output_dir, f"gt_{i+1:02d}_{img_name}")
        cv2.imwrite(output_path, annotated_image)
        
        # Copy original image and label
        shutil.copy2(img_path, os.path.join(output_dir, f"original_{i+1:02d}_{img_name}"))
        shutil.copy2(label_path, os.path.join(output_dir, f"original_{i+1:02d}_{img_path.stem}.txt"))
        
        # Create summary
        class_summary = ", ".join([f"{class_names[i]}: {count}" for i, count in enumerate(class_counts) if count > 0])
        results.append({
            "index": i+1,
            "filename": img_name,
            "class_counts": class_counts,
            "class_summary": class_summary,
            "total_objects": sum(class_counts),
            "score": score
        })
        
        print(f"Selected image {i+1}: {img_name} - {class_summary}")
    
    # Write summary file
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        for res in results:
            f.write(f"Image {res['index']}: {res['filename']}\n")
            f.write(f"  Classes: {res['class_summary']}\n")
            f.write(f"  Total objects: {res['total_objects']}\n")
            f.write("\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Select ground truth images with annotations")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save selected images")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to select")
    
    args = parser.parse_args()
    
    select_gt_images(args.dataset_path, args.output_dir, args.num_images)

if __name__ == "__main__":
    main()
