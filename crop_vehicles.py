#!/usr/bin/env python3
"""
Script to crop vehicles with extra padding and maintain relative positions of other objects.
Creates standardized 640x640 training images focused on vehicles.
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import json
from tqdm import tqdm

def load_classes(yaml_path):
    """Load class names from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def get_vehicle_bbox(annotations, class_names, img_width, img_height):
    """
    Get the bounding box of the vehicle with the largest area.
    Returns normalized coordinates (xmin, ymin, xmax, ymax).
    """
    vehicle_boxes = []
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if cls_id < len(class_names) and class_names[cls_id] == 'vehicle':
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to absolute coordinates
                abs_width = width * img_width
                abs_height = height * img_height
                area = abs_width * abs_height
                
                # Convert to corner coordinates
                xmin = x_center - width/2
                ymin = y_center - height/2
                xmax = x_center + width/2
                ymax = y_center + height/2
                
                vehicle_boxes.append((xmin, ymin, xmax, ymax, area))
    
    if not vehicle_boxes:
        return None
        
    # Get the vehicle with the largest area
    vehicle_boxes.sort(key=lambda x: x[4], reverse=True)
    return vehicle_boxes[0][:4]  # Return (xmin, ymin, xmax, ymax)

def add_padding(bbox, padding_ratio=0.2, img_width=1, img_height=1):
    """
    Add padding around the bounding box.
    Returns normalized coordinates (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = bbox
    
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate padding
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    
    # Add padding
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(1, xmax + pad_x)
    ymax = min(1, ymax + pad_y)
    
    return (xmin, ymin, xmax, ymax)

def make_square(bbox, img_width, img_height):
    """
    Make the bounding box square while maintaining aspect ratio.
    Returns normalized coordinates (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Convert to absolute coordinates
    xmin_abs = xmin * img_width
    ymin_abs = ymin * img_height
    xmax_abs = xmax * img_width
    ymax_abs = ymax * img_height
    
    width = xmax_abs - xmin_abs
    height = ymax_abs - ymin_abs
    
    # Make square
    if width > height:
        # Add to height
        diff = width - height
        ymin_abs -= diff / 2
        ymax_abs += diff / 2
    else:
        # Add to width
        diff = height - width
        xmin_abs -= diff / 2
        xmax_abs += diff / 2
    
    # Ensure within image bounds
    xmin_abs = max(0, xmin_abs)
    ymin_abs = max(0, ymin_abs)
    xmax_abs = min(img_width, xmax_abs)
    ymax_abs = min(img_height, ymax_abs)
    
    # Convert back to normalized coordinates
    xmin = xmin_abs / img_width
    ymin = ymin_abs / img_height
    xmax = xmax_abs / img_width
    ymax = ymax_abs / img_height
    
    return (xmin, ymin, xmax, ymax)

def transform_annotations(annotations, crop_bbox, class_names, img_width, img_height, min_logo_carplate_size=20, top_margin_percent=20):
    """
    Transform annotations to be relative to the cropped bounding box.
    Returns a list of transformed annotations.
    
    Parameters:
    - min_logo_carplate_size: Minimum size in pixels for logos and carplates
    - top_margin_percent: Percentage of the top of the image to apply stricter filtering
    """
    transformed_annotations = []
    
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_bbox
    crop_width = crop_xmax - crop_xmin
    crop_height = crop_ymax - crop_ymin
    
    # Calculate the top margin boundary
    top_margin_y = crop_ymin + (crop_height * top_margin_percent / 100)
    
    # Find the main vehicle (largest vehicle) in the cropped area
    main_vehicle_center = None
    main_vehicle_size = 0
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if cls_id < len(class_names) and class_names[cls_id] == 'vehicle':
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check if the vehicle is within the crop area
                if (x_center >= crop_xmin and x_center <= crop_xmax and 
                    y_center >= crop_ymin and y_center <= crop_ymax):
                    # Calculate absolute size
                    abs_width = width * img_width
                    abs_height = height * img_height
                    vehicle_size = abs_width * abs_height
                    
                    if vehicle_size > main_vehicle_size:
                        main_vehicle_size = vehicle_size
                        main_vehicle_center = (x_center, y_center)
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Get class name
            cls_name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
            
            # Check if the annotation is within the crop area
            ann_xmin = x_center - width/2
            ann_ymin = y_center - height/2
            ann_xmax = x_center + width/2
            ann_ymax = y_center + height/2
            
            # Calculate absolute size
            abs_width = width * img_width
            abs_height = height * img_height
            
            # Calculate intersection with crop area
            int_xmin = max(crop_xmin, ann_xmin)
            int_ymin = max(crop_ymin, ann_ymin)
            int_xmax = min(crop_xmax, ann_xmax)
            int_ymax = min(crop_ymax, ann_ymax)
            
            # Check if there is an intersection
            if int_xmax > int_xmin and int_ymax > int_ymin:
                # Calculate intersection area
                int_width = int_xmax - int_xmin
                int_height = int_ymax - int_ymin
                int_area = int_width * int_height
                
                # Calculate annotation area
                ann_area = width * height
                
                # Calculate overlap ratio
                overlap_ratio = int_area / ann_area
                
                # Apply additional filtering based on class and position
                should_keep = False
                
                # For logos and carplates, apply size threshold
                if cls_name in ['logo', 'carplate']:
                    # Check if the object is large enough
                    if abs_width >= min_logo_carplate_size and abs_height >= min_logo_carplate_size:
                        # If in top margin, apply stricter rules
                        if y_center < top_margin_y:
                            # Only keep if it's close to the main vehicle or very large
                            if main_vehicle_center:
                                dist_to_main = ((x_center - main_vehicle_center[0])**2 + 
                                               (y_center - main_vehicle_center[1])**2)**0.5
                                # Keep if close to main vehicle or very large
                                if dist_to_main < 0.2 or (abs_width > 40 and abs_height > 40):
                                    should_keep = overlap_ratio >= 0.7  # Stricter overlap requirement
                        else:
                            # Not in top margin, use normal overlap threshold
                            should_keep = overlap_ratio >= 0.5
                elif cls_name == 'vehicle':
                    # For vehicles, keep main vehicle and those with good overlap
                    if main_vehicle_center and (x_center, y_center) == main_vehicle_center:
                        should_keep = True  # Always keep main vehicle
                    else:
                        # For other vehicles, apply stricter filtering in top margin
                        if y_center < top_margin_y:
                            should_keep = overlap_ratio >= 0.8 and abs_width >= 50 and abs_height >= 50
                        else:
                            should_keep = overlap_ratio >= 0.5
                elif cls_name == 'motorcycle':
                    # For motorcycles, similar rules as vehicles
                    if y_center < top_margin_y:
                        should_keep = overlap_ratio >= 0.8 and abs_width >= 50 and abs_height >= 50
                    else:
                        should_keep = overlap_ratio >= 0.5
                
                if should_keep:
                    # Transform coordinates to be relative to the crop
                    new_x_center = (x_center - crop_xmin) / crop_width
                    new_y_center = (y_center - crop_ymin) / crop_height
                    new_width = width / crop_width
                    new_height = height / crop_height
                    
                    # Ensure the transformed annotation is within [0,1]
                    new_x_center = max(0, min(1, new_x_center))
                    new_y_center = max(0, min(1, new_y_center))
                    new_width = min(new_width, 2*new_x_center, 2*(1-new_x_center))
                    new_height = min(new_height, 2*new_y_center, 2*(1-new_y_center))
                    
                    # Add to transformed annotations
                    transformed_annotations.append(f"{cls_id} {new_x_center} {new_y_center} {new_width} {new_height}")
    
    return transformed_annotations

def crop_and_resize(image, crop_bbox, target_size=640):
    """
    Crop the image using the bounding box and resize to target_size x target_size.
    """
    img_height, img_width = image.shape[:2]
    
    # Convert normalized coordinates to absolute
    xmin = int(crop_bbox[0] * img_width)
    ymin = int(crop_bbox[1] * img_height)
    xmax = int(crop_bbox[2] * img_width)
    ymax = int(crop_bbox[3] * img_height)
    
    # Crop the image
    cropped = image[ymin:ymax, xmin:xmax]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized

def process_dataset(dataset_path, output_dir, target_size=640, padding_ratio=0.2, min_logo_carplate_size=20, top_margin_percent=20):
    """
    Process a dataset to crop vehicles with padding and transform annotations.
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")
    
    # Load class names
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = load_classes(yaml_path)
    
    # Setup output directories
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    images_output_dir = os.path.join(dataset_output_dir, "images")
    labels_output_dir = os.path.join(dataset_output_dir, "labels")
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Copy data.yaml
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    yaml_data['path'] = dataset_output_dir
    
    with open(os.path.join(dataset_output_dir, "data.yaml"), 'w') as f:
        yaml.dump(yaml_data, f)
    
    # Find all image paths
    image_dir = os.path.join(dataset_path, "train", "images")
    label_dir = os.path.join(dataset_path, "train", "labels")
    
    image_paths = list(Path(image_dir).glob("*.jpg"))
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_paths, desc=f"Processing {dataset_name}"):
        label_path = os.path.join(label_dir, img_path.with_suffix(".txt").name)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            skipped_count += 1
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            skipped_count += 1
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        # Get vehicle bounding box
        vehicle_bbox = get_vehicle_bbox(annotations, class_names, img_width, img_height)
        if vehicle_bbox is None:
            skipped_count += 1
            continue
        
        # Add padding
        padded_bbox = add_padding(vehicle_bbox, padding_ratio, img_width, img_height)
        
        # Make square
        square_bbox = make_square(padded_bbox, img_width, img_height)
        
        # Transform annotations
        transformed_annotations = transform_annotations(annotations, square_bbox, class_names, img_width, img_height, min_logo_carplate_size, top_margin_percent)
        
        # Crop and resize image
        cropped_image = crop_and_resize(image, square_bbox, target_size)
        
        # Save cropped image
        output_img_path = os.path.join(images_output_dir, img_path.name)
        cv2.imwrite(output_img_path, cropped_image)
        
        # Save transformed annotations
        output_label_path = os.path.join(labels_output_dir, img_path.with_suffix(".txt").name)
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(transformed_annotations))
        
        processed_count += 1
    
    print(f"Dataset {dataset_name}: Processed {processed_count} images, Skipped {skipped_count} images")
    return processed_count, skipped_count

def visualize_samples(output_dir, num_samples=5):
    """
    Create visualization of sample processed images with bounding boxes.
    """
    html_path = os.path.join(output_dir, "crop_samples.html")
    
    # Find all dataset directories
    dataset_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    samples = []
    
    for dataset in dataset_dirs:
        dataset_path = os.path.join(output_dir, dataset)
        
        # Load class names
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            continue
            
        class_names = load_classes(yaml_path)
        
        # Find image and label paths
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue
            
        image_paths = list(Path(images_dir).glob("*.jpg"))
        
        # Select random samples
        if len(image_paths) > num_samples:
            import random
            image_paths = random.sample(image_paths, num_samples)
        
        # Process each sample
        for img_path in image_paths:
            label_path = os.path.join(labels_dir, img_path.with_suffix(".txt").name)
            
            if not os.path.exists(label_path):
                continue
                
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Load annotations
            with open(label_path, 'r') as f:
                annotations = f.readlines()
                
            # Draw bounding boxes
            image_with_boxes = image.copy()
            
            class_colors = {
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
                    color = class_colors.get(cls_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{cls_name}"
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image_with_boxes, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Save visualization
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = os.path.join(vis_dir, f"{dataset}_{img_path.name}")
            cv2.imwrite(vis_path, image_with_boxes)
            
            # Add to samples
            samples.append({
                'dataset': dataset,
                'image_path': str(img_path),
                'vis_path': vis_path,
                'num_annotations': len(annotations)
            })
    
    # Generate HTML
    with open(html_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Crop Samples</title>
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
        .sample-container {
            margin-bottom: 30px;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .sample-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .sample-metadata {
            margin-bottom: 10px;
            font-size: 14px;
            color: #555;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
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
    </style>
</head>
<body>
    <h1>Vehicle Crop Samples (640x640)</h1>
    
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
    
    <h2>Sample Cropped Images</h2>
""")

        # Add samples
        for i, sample in enumerate(samples):
            dataset = sample['dataset']
            image_path = sample['image_path']
            vis_path = os.path.basename(sample['vis_path'])
            num_annotations = sample['num_annotations']
            
            f.write(f'    <div class="sample-container">\n')
            f.write(f'        <div class="sample-title">Sample {i+1}: {os.path.basename(image_path)}</div>\n')
            f.write(f'        <div class="sample-metadata">Dataset: {dataset}</div>\n')
            f.write(f'        <div class="sample-metadata">Annotations: {num_annotations}</div>\n')
            f.write(f'        <img src="visualizations/{vis_path}" alt="Sample {i+1}">\n')
            f.write(f'    </div>\n')
            
        f.write("""</body>
</html>""")
    
    print(f"Sample visualizations created at {html_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop vehicles with padding and transform annotations")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Paths to dataset directories")
    parser.add_argument("--output_dir", default="cropped_vehicles", help="Directory to save processed datasets")
    parser.add_argument("--target_size", type=int, default=640, help="Target size for cropped images")
    parser.add_argument("--padding_ratio", type=float, default=0.2, help="Padding ratio around vehicle")
    parser.add_argument("--min_logo_carplate_size", type=int, default=20, help="Minimum size in pixels for logos and carplates")
    parser.add_argument("--top_margin_percent", type=int, default=20, help="Percentage of the top of the image to apply stricter filtering")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sample visualizations per dataset")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    for dataset_path in args.datasets:
        processed, skipped = process_dataset(
            dataset_path, 
            args.output_dir, 
            args.target_size, 
            args.padding_ratio,
            args.min_logo_carplate_size,
            args.top_margin_percent
        )
        total_processed += processed
        total_skipped += skipped
    
    print(f"Total: Processed {total_processed} images, Skipped {total_skipped} images")
    
    # Create sample visualizations
    visualize_samples(args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
