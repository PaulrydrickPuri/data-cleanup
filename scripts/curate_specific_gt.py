#!/usr/bin/env python3
"""
Script to curate ground truth images with specific class combinations:
1. Images with both carplate and vehicle
2. Images with both motorcycle and carplate
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import shutil
import json

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

def filter_annotations(annotations, img_width, img_height, min_size=64, max_overlap_ratio=0.5):
    """
    Filter out problematic annotations:
    - Too small
    - Too close to edge
    - Overlapping too much
    """
    filtered_annotations = []
    boxes = []
    
    # First pass: filter by size and edge proximity
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to absolute coordinates
            abs_width = width * img_width
            abs_height = height * img_height
            
            # Check if box is too small
            if abs_width < min_size or abs_height < min_size:
                continue
                
            # Check if box is too close to edge
            edge_margin = 0.02  # 2% of image dimension
            if (x_center - width/2 < edge_margin or 
                x_center + width/2 > 1 - edge_margin or
                y_center - height/2 < edge_margin or
                y_center + height/2 > 1 - edge_margin):
                continue
                
            # Add to filtered list
            filtered_annotations.append(ann)
            boxes.append((x_center, y_center, width, height))
    
    # Second pass: filter overlapping boxes
    final_annotations = []
    final_boxes = []
    
    for i, (ann, box) in enumerate(zip(filtered_annotations, boxes)):
        x1, y1, w1, h1 = box
        
        # Check overlap with already accepted boxes
        overlap_too_much = False
        for j, (x2, y2, w2, h2) in enumerate(final_boxes):
            # Calculate intersection
            x_overlap = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
            y_overlap = max(0, min(y1 + h1/2, y2 + h2/2) - max(y1 - h1/2, y2 - h2/2))
            intersection = x_overlap * y_overlap
            
            # Calculate areas
            area1 = w1 * h1
            area2 = w2 * h2
            
            # Calculate overlap ratio
            overlap_ratio = intersection / min(area1, area2)
            
            if overlap_ratio > max_overlap_ratio:
                overlap_too_much = True
                break
                
        if not overlap_too_much:
            final_annotations.append(ann)
            final_boxes.append(box)
    
    return final_annotations

def check_class_combinations(annotations, class_names):
    """
    Check if the image contains specific class combinations:
    1. Both carplate and vehicle
    2. Both motorcycle and carplate
    """
    classes_present = set()
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if 0 <= cls_id < len(class_names):
                cls_name = class_names[cls_id]
                classes_present.add(cls_name)
    
    has_vehicle_and_carplate = 'vehicle' in classes_present and 'carplate' in classes_present
    has_motorcycle_and_carplate = 'motorcycle' in classes_present and 'carplate' in classes_present
    
    return {
        'vehicle_and_carplate': has_vehicle_and_carplate,
        'motorcycle_and_carplate': has_motorcycle_and_carplate,
        'classes_present': classes_present
    }

def evaluate_image_quality(image_path, label_path, class_names):
    """
    Evaluate the quality of an image for ground truth selection.
    Returns a score and quality metrics.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, {"error": "Failed to load image"}, None, None
        
    img_height, img_width = image.shape[:2]
    
    # Check if label file exists
    if not os.path.exists(label_path):
        return 0, {"error": "No label file found"}, None, None
        
    # Load annotations
    with open(label_path, 'r') as f:
        annotations = f.readlines()
        
    if not annotations:
        return 0, {"error": "No annotations found"}, None, None
        
    # Filter annotations
    filtered_annotations = filter_annotations(annotations, img_width, img_height)
    
    # Check class combinations
    class_check = check_class_combinations(filtered_annotations, class_names)
    
    # Calculate image quality metrics
    # 1. Blur detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Exposure check
    mean_brightness = np.mean(gray)
    
    # Count classes
    class_counts = {}
    for ann in filtered_annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cls_name = class_names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    # Calculate final score
    # Weight factors:
    # - Sharpness (higher is better)
    # - Good exposure (closer to 128 is better)
    # - Has required class combinations (huge bonus)
    
    sharpness_score = min(laplacian_var / 500, 1.0) * 30  # Max 30 points
    exposure_score = (1 - abs(mean_brightness - 128) / 128) * 20  # Max 20 points
    
    # Bonus for having required class combinations
    combination_score = 0
    if class_check['vehicle_and_carplate']:
        combination_score += 50  # Big bonus for vehicle+carplate
    if class_check['motorcycle_and_carplate']:
        combination_score += 70  # Even bigger bonus for motorcycle+carplate (rarer)
    
    total_score = sharpness_score + exposure_score + combination_score
    
    metrics = {
        "sharpness": laplacian_var,
        "sharpness_score": sharpness_score,
        "brightness": mean_brightness,
        "exposure_score": exposure_score,
        "combination_score": combination_score,
        "vehicle_and_carplate": class_check['vehicle_and_carplate'],
        "motorcycle_and_carplate": class_check['motorcycle_and_carplate'],
        "total_score": total_score,
        "class_counts": class_counts
    }
    
    return total_score, metrics, filtered_annotations, image

def curate_specific_gt_images(dataset_paths, output_dir, 
                             vehicle_carplate_count=10, 
                             motorcycle_carplate_count=10):
    """
    Curate ground truth images with specific class combinations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for each combination
    vehicle_carplate_dir = os.path.join(output_dir, "vehicle_carplate")
    motorcycle_carplate_dir = os.path.join(output_dir, "motorcycle_carplate")
    
    os.makedirs(vehicle_carplate_dir, exist_ok=True)
    os.makedirs(motorcycle_carplate_dir, exist_ok=True)
    
    # Lists to store selected images
    vehicle_carplate_images = []
    motorcycle_carplate_images = []
    
    # Process all datasets
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        dataset_name = os.path.basename(dataset_path)
        print(f"Processing dataset {dataset_idx+1}: {dataset_name}")
        
        # Load class names
        yaml_path = os.path.join(dataset_path, "data.yaml")
        class_names = load_classes(yaml_path)
        
        # Find all image paths
        image_dir = os.path.join(dataset_path, "train", "images")
        label_dir = os.path.join(dataset_path, "train", "labels")
        
        image_paths = list(Path(image_dir).glob("*.jpg"))
        
        # Evaluate all images
        for img_path in image_paths:
            label_path = os.path.join(label_dir, img_path.with_suffix(".txt").name)
            
            score, metrics, filtered_annotations, image = evaluate_image_quality(img_path, label_path, class_names)
            
            if score <= 0 or filtered_annotations is None or image is None:
                continue
                
            # Check if image has required class combinations
            if metrics['vehicle_and_carplate'] and len(vehicle_carplate_images) < vehicle_carplate_count:
                vehicle_carplate_images.append({
                    'dataset': dataset_name,
                    'dataset_idx': dataset_idx,
                    'img_path': img_path,
                    'score': score,
                    'metrics': metrics,
                    'filtered_annotations': filtered_annotations,
                    'class_names': class_names,
                    'image': image
                })
                
            if metrics['motorcycle_and_carplate'] and len(motorcycle_carplate_images) < motorcycle_carplate_count:
                motorcycle_carplate_images.append({
                    'dataset': dataset_name,
                    'dataset_idx': dataset_idx,
                    'img_path': img_path,
                    'score': score,
                    'metrics': metrics,
                    'filtered_annotations': filtered_annotations,
                    'class_names': class_names,
                    'image': image
                })
                
        # Sort by score (descending)
        vehicle_carplate_images.sort(key=lambda x: x['score'], reverse=True)
        motorcycle_carplate_images.sort(key=lambda x: x['score'], reverse=True)
        
        # Trim to desired count
        vehicle_carplate_images = vehicle_carplate_images[:vehicle_carplate_count]
        motorcycle_carplate_images = motorcycle_carplate_images[:motorcycle_carplate_count]
        
        print(f"Found {len(vehicle_carplate_images)} vehicle+carplate images so far")
        print(f"Found {len(motorcycle_carplate_images)} motorcycle+carplate images so far")
    
    # Process and save vehicle+carplate images
    process_and_save_images(vehicle_carplate_images, vehicle_carplate_dir, "vehicle_carplate")
    
    # Process and save motorcycle+carplate images
    process_and_save_images(motorcycle_carplate_images, motorcycle_carplate_dir, "motorcycle_carplate")
    
    # Create HTML viewer
    create_html_viewer(output_dir, vehicle_carplate_images, motorcycle_carplate_images)
    
    return {
        'vehicle_carplate': vehicle_carplate_images,
        'motorcycle_carplate': motorcycle_carplate_images
    }

def process_and_save_images(selected_images, output_dir, prefix):
    """Process and save selected images with their annotations."""
    for i, item in enumerate(selected_images):
        img_path = item['img_path']
        img_name = img_path.name
        filtered_annotations = item['filtered_annotations']
        class_names = item['class_names']
        image = item['image']
        
        img_height, img_width = image.shape[:2]
        
        # Draw annotations
        annotated_image = draw_annotations(image.copy(), filtered_annotations, img_width, img_height, class_names)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"{prefix}_{i+1:02d}_{img_name}")
        cv2.imwrite(output_path, annotated_image)
        
        # Save filtered annotations
        filtered_label_path = os.path.join(output_dir, f"{prefix}_{i+1:02d}_{img_path.stem}.txt")
        with open(filtered_label_path, 'w') as f:
            f.writelines(filtered_annotations)
            
        # Save original image
        original_path = os.path.join(output_dir, f"original_{i+1:02d}_{img_name}")
        shutil.copy2(img_path, original_path)
        
        # Create summary
        class_counts = {}
        for ann in filtered_annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cls_name = class_names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
        class_summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        
        print(f"Saved {prefix} image {i+1}: {img_name} - {class_summary} (Score: {item['score']:.2f})")

def create_html_viewer(output_dir, vehicle_carplate_images, motorcycle_carplate_images):
    """Create HTML viewer for the curated ground truth images."""
    html_path = os.path.join(output_dir, "specific_gt_viewer.html")
    
    # Generate HTML
    with open(html_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Specific Ground Truth Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        .dataset-container {
            margin-bottom: 40px;
        }
        .image-container {
            margin-bottom: 30px;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .image-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .image-metadata {
            margin-bottom: 10px;
            font-size: 14px;
            color: #555;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            background-color: #e0e0e0;
            cursor: pointer;
            border: 1px solid #ccc;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .tab.active {
            background-color: #fff;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
            z-index: 1;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .legend {
            margin-bottom: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
        }
        .metrics {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Specific Ground Truth Images</h1>
    
    <div class="legend">
        <h3>Color Legend:</h3>
        <div class="legend-item">
            <span class="color-box" style="background-color: rgb(0, 255, 0);"></span>
            <span>Vehicle</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background-color: rgb(255, 0, 0);"></span>
            <span>Carplate</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background-color: rgb(0, 0, 255);"></span>
            <span>Motorcycle</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background-color: rgb(255, 255, 0);"></span>
            <span>Logo</span>
        </div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('vehicle_carplate')">Vehicle + Carplate</div>
        <div class="tab" onclick="switchTab('motorcycle_carplate')">Motorcycle + Carplate</div>
    </div>
    
    <div id="vehicle_carplate" class="tab-content active">
        <h2>Vehicle + Carplate Images</h2>
        
        <div class="dataset-container">
""")

        # Generate vehicle+carplate images
        for i, item in enumerate(vehicle_carplate_images):
            img_path = item['img_path']
            img_name = img_path.name
            dataset_name = item['dataset']
            score = item['score']
            metrics = item['metrics']
            
            class_counts = metrics['class_counts']
            class_summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            
            f.write(f'            <!-- Image {i+1} -->\n')
            f.write('            <div class="image-container">\n')
            f.write(f'                <div class="image-title">Image {i+1}: {img_name}</div>\n')
            f.write(f'                <div class="image-metadata">Dataset: {dataset_name}</div>\n')
            f.write(f'                <div class="image-metadata">Classes: {class_summary}</div>\n')
            f.write(f'                <div class="metrics">Quality Score: {score:.2f}</div>\n')
            f.write(f'                <img src="vehicle_carplate/vehicle_carplate_{i+1:02d}_{img_name}" alt="Vehicle+Carplate Image {i+1}">\n')
            f.write('            </div>\n')
            
        f.write("""        </div>
    </div>
    
    <div id="motorcycle_carplate" class="tab-content">
        <h2>Motorcycle + Carplate Images</h2>
        
        <div class="dataset-container">
""")

        # Generate motorcycle+carplate images
        for i, item in enumerate(motorcycle_carplate_images):
            img_path = item['img_path']
            img_name = img_path.name
            dataset_name = item['dataset']
            score = item['score']
            metrics = item['metrics']
            
            class_counts = metrics['class_counts']
            class_summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            
            f.write(f'            <!-- Image {i+1} -->\n')
            f.write('            <div class="image-container">\n')
            f.write(f'                <div class="image-title">Image {i+1}: {img_name}</div>\n')
            f.write(f'                <div class="image-metadata">Dataset: {dataset_name}</div>\n')
            f.write(f'                <div class="image-metadata">Classes: {class_summary}</div>\n')
            f.write(f'                <div class="metrics">Quality Score: {score:.2f}</div>\n')
            f.write(f'                <img src="motorcycle_carplate/motorcycle_carplate_{i+1:02d}_{img_name}" alt="Motorcycle+Carplate Image {i+1}">\n')
            f.write('            </div>\n')
            
        f.write("""        </div>
    </div>
    
    <script>
        function switchTab(tabId) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            // Deactivate all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Activate selected tab and content
            document.getElementById(tabId).classList.add('active');
            document.querySelector(`.tab[onclick="switchTab('${tabId}')"]`).classList.add('active');
        }
    </script>
</body>
</html>""")
    
    print(f"HTML viewer created at {html_path}")

def main():
    parser = argparse.ArgumentParser(description="Curate ground truth images with specific class combinations")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Paths to dataset directories")
    parser.add_argument("--output_dir", default="specific_gt", help="Directory to save selected images")
    parser.add_argument("--vehicle_carplate_count", type=int, default=10, help="Number of vehicle+carplate images to select")
    parser.add_argument("--motorcycle_carplate_count", type=int, default=10, help="Number of motorcycle+carplate images to select")
    
    args = parser.parse_args()
    
    curate_specific_gt_images(
        args.datasets, 
        args.output_dir, 
        args.vehicle_carplate_count, 
        args.motorcycle_carplate_count
    )

if __name__ == "__main__":
    main()
