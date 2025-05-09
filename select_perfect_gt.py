#!/usr/bin/env python3
"""
Script to select perfect ground truth images and clean up annotations.
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

def evaluate_image_quality(image_path, label_path, class_names):
    """
    Evaluate the quality of an image for ground truth selection.
    Returns a score and quality metrics.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, {"error": "Failed to load image"}
        
    img_height, img_width = image.shape[:2]
    
    # Check if label file exists
    if not os.path.exists(label_path):
        return 0, {"error": "No label file found"}
        
    # Load annotations
    with open(label_path, 'r') as f:
        annotations = f.readlines()
        
    if not annotations:
        return 0, {"error": "No annotations found"}
        
    # Count classes
    class_counts = [0] * len(class_names)
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if 0 <= cls_id < len(class_names):
                class_counts[cls_id] += 1
                
    # Calculate image quality metrics
    # 1. Blur detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Exposure check
    mean_brightness = np.mean(gray)
    
    # 3. Class distribution
    num_classes_present = sum(1 for count in class_counts if count > 0)
    total_objects = sum(class_counts)
    
    # Filter annotations
    filtered_annotations = filter_annotations(annotations, img_width, img_height)
    num_filtered = len(annotations) - len(filtered_annotations)
    
    # Calculate final score
    # Weight factors:
    # - Sharpness (higher is better)
    # - Good exposure (closer to 128 is better)
    # - Class diversity (more classes is better)
    # - Number of objects (more is better, up to a point)
    # - Percentage of good annotations (higher is better)
    
    sharpness_score = min(laplacian_var / 500, 1.0) * 30  # Max 30 points
    exposure_score = (1 - abs(mean_brightness - 128) / 128) * 20  # Max 20 points
    class_diversity_score = (num_classes_present / len(class_names)) * 30  # Max 30 points
    objects_score = min(total_objects / 10, 1.0) * 10  # Max 10 points
    annotation_quality_score = (len(filtered_annotations) / max(1, len(annotations))) * 10  # Max 10 points
    
    total_score = sharpness_score + exposure_score + class_diversity_score + objects_score + annotation_quality_score
    
    metrics = {
        "sharpness": laplacian_var,
        "sharpness_score": sharpness_score,
        "brightness": mean_brightness,
        "exposure_score": exposure_score,
        "classes_present": num_classes_present,
        "class_diversity_score": class_diversity_score,
        "total_objects": total_objects,
        "objects_score": objects_score,
        "annotations_before": len(annotations),
        "annotations_after": len(filtered_annotations),
        "annotation_quality_score": annotation_quality_score,
        "total_score": total_score,
        "class_counts": {class_names[i]: count for i, count in enumerate(class_counts) if count > 0}
    }
    
    return total_score, metrics, filtered_annotations

def select_perfect_gt_images(dataset_paths, output_dir, num_images_per_dataset=5):
    """
    Select perfect ground truth images from multiple datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_selected = []
    
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
        image_scores = []
        for img_path in image_paths:
            label_path = os.path.join(label_dir, img_path.with_suffix(".txt").name)
            score, metrics, filtered_annotations = evaluate_image_quality(img_path, label_path, class_names)
            
            if score > 0:  # Only consider images with valid scores
                image_scores.append((img_path, score, metrics, filtered_annotations))
                
        # Sort by score (descending)
        image_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top images
        selected_images = image_scores[:num_images_per_dataset]
        
        # Process selected images
        dataset_output_dir = os.path.join(output_dir, f"dataset{dataset_idx+1}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        dataset_results = []
        for i, (img_path, score, metrics, filtered_annotations) in enumerate(selected_images):
            img_name = img_path.name
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load image {img_name}")
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Draw annotations
            annotated_image = draw_annotations(image.copy(), filtered_annotations, img_width, img_height, class_names)
            
            # Save annotated image
            output_path = os.path.join(dataset_output_dir, f"gt_{i+1:02d}_{img_name}")
            cv2.imwrite(output_path, annotated_image)
            
            # Save filtered annotations
            filtered_label_path = os.path.join(dataset_output_dir, f"gt_{i+1:02d}_{img_path.stem}.txt")
            with open(filtered_label_path, 'w') as f:
                f.writelines(filtered_annotations)
                
            # Save original image
            original_path = os.path.join(dataset_output_dir, f"original_{i+1:02d}_{img_name}")
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
            
            result = {
                "dataset": dataset_name,
                "index": i+1,
                "filename": img_name,
                "class_counts": class_counts,
                "class_summary": class_summary,
                "total_objects": sum(class_counts.values()),
                "score": score,
                "metrics": metrics
            }
            
            dataset_results.append(result)
            all_selected.append(result)
            
            print(f"Selected image {i+1}: {img_name} - {class_summary} (Score: {score:.2f})")
            
        # Save dataset summary
        with open(os.path.join(dataset_output_dir, "summary.json"), 'w') as f:
            json.dump(dataset_results, f, indent=2)
            
    # Save overall summary
    with open(os.path.join(output_dir, "all_selected.json"), 'w') as f:
        json.dump(all_selected, f, indent=2)
        
    # Create HTML viewer
    create_html_viewer(output_dir, all_selected)
    
    return all_selected

def create_html_viewer(output_dir, all_selected):
    """Create HTML viewer for the selected ground truth images."""
    html_path = os.path.join(output_dir, "perfect_gt_viewer.html")
    
    # Group by dataset
    datasets = {}
    for item in all_selected:
        dataset = item["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(item)
        
    # Generate HTML
    with open(html_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perfect Ground Truth Viewer</title>
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
    <h1>Perfect Ground Truth Images with Clean Annotations</h1>
    
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
""")

        # Generate tabs
        for i, dataset in enumerate(datasets.keys()):
            active = ' active' if i == 0 else ''
            f.write(f'        <div class="tab{active}" onclick="switchTab(\'dataset{i+1}\')">{dataset}</div>\n')
            
        f.write("""    </div>
    
""")

        # Generate tab content
        for i, (dataset, items) in enumerate(datasets.items()):
            active = ' active' if i == 0 else ''
            f.write(f'    <div id="dataset{i+1}" class="tab-content{active}">\n')
            f.write(f'        <h2>Dataset {i+1}: {dataset}</h2>\n')
            f.write('        \n        <div class="dataset-container">\n')
            
            # Generate image containers
            for item in items:
                idx = item["index"]
                filename = item["filename"]
                class_summary = item["class_summary"]
                score = item["score"]
                
                f.write(f'            <!-- Image {idx} -->\n')
                f.write('            <div class="image-container">\n')
                f.write(f'                <div class="image-title">Image {idx}: {filename}</div>\n')
                f.write(f'                <div class="image-metadata">Classes: {class_summary}</div>\n')
                f.write(f'                <div class="metrics">Quality Score: {score:.2f}</div>\n')
                f.write(f'                <img src="dataset{i+1}/gt_{idx:02d}_{filename}" alt="Ground Truth Image {idx}">\n')
                f.write('            </div>\n')
                
            f.write('        </div>\n')
            f.write('    </div>\n')
            
        f.write("""    
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
    parser = argparse.ArgumentParser(description="Select perfect ground truth images with clean annotations")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Paths to dataset directories")
    parser.add_argument("--output_dir", default="perfect_gt", help="Directory to save selected images")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to select per dataset")
    
    args = parser.parse_args()
    
    select_perfect_gt_images(args.datasets, args.output_dir, args.num_images)

if __name__ == "__main__":
    main()
