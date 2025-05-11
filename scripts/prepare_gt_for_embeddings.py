#!/usr/bin/env python3
"""
Script to prepare ground truth images and labels for CLIP embedding generation.
This script:
1. Extracts all ground truth images from the HTML viewer
2. Creates a text file with paths to these images
3. Creates a text file with corresponding class labels
"""
import os
import re
import argparse
import shutil
from pathlib import Path
import yaml
import json
from bs4 import BeautifulSoup

def extract_gt_images_from_html(html_path, output_dir):
    """
    Extract ground truth images referenced in the HTML viewer and create
    a mapping of image paths to their class labels.
    """
    # Read HTML file
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image containers
    image_containers = soup.find_all('div', class_='image-container')
    
    image_paths = []
    image_classes = {}
    
    for container in image_containers:
        # Get image metadata
        metadata = container.find('div', class_='image-metadata')
        if metadata:
            classes_text = metadata.text.strip()
            # Extract classes using regex
            classes_match = re.search(r'Classes:\s*(.*)', classes_text)
            if classes_match:
                classes_str = classes_match.group(1)
                # Parse classes like "vehicle: 1, carplate: 1, logo: 1"
                classes = {}
                for class_item in classes_str.split(','):
                    if ':' in class_item:
                        cls_name, cls_count = class_item.split(':')
                        cls_name = cls_name.strip()
                        cls_count = int(cls_count.strip())
                        classes[cls_name] = cls_count
        
        # Get image
        img_tag = container.find('img')
        if img_tag and 'src' in img_tag.attrs:
            img_src = img_tag['src']
            img_path = os.path.join(os.path.dirname(html_path), img_src)
            
            # Copy image to output directory
            if os.path.exists(img_path):
                img_name = os.path.basename(img_path)
                output_path = os.path.join(output_dir, img_name)
                shutil.copy2(img_path, output_path)
                
                # Add to list
                image_paths.append(output_path)
                image_classes[output_path] = classes
    
    return image_paths, image_classes

def create_class_specific_lists(image_paths, image_classes, output_dir):
    """
    Create text files with image paths and class labels for each class.
    """
    # Initialize class-specific lists
    class_images = {
        'vehicle': [],
        'carplate': [],
        'motorcycle': [],
        'logo': []
    }
    
    # Group images by class
    for img_path, classes in image_classes.items():
        for cls_name, cls_count in classes.items():
            if cls_name in class_images and cls_count > 0:
                class_images[cls_name].append(img_path)
    
    # Create output files
    for cls_name, img_list in class_images.items():
        if img_list:
            # Images file
            images_path = os.path.join(output_dir, f"{cls_name}_images.txt")
            with open(images_path, 'w') as f:
                f.write('\n'.join(img_list))
            
            # Labels file
            labels_path = os.path.join(output_dir, f"{cls_name}_labels.txt")
            with open(labels_path, 'w') as f:
                f.write('\n'.join([cls_name] * len(img_list)))
            
            print(f"Created {cls_name} lists with {len(img_list)} images")
    
    # Create combined lists
    all_images = []
    all_labels = []
    
    for cls_name, img_list in class_images.items():
        for img_path in img_list:
            all_images.append(img_path)
            all_labels.append(cls_name)
    
    # Write combined lists
    all_images_path = os.path.join(output_dir, "all_gt_images.txt")
    with open(all_images_path, 'w') as f:
        f.write('\n'.join(all_images))
    
    all_labels_path = os.path.join(output_dir, "all_gt_labels.txt")
    with open(all_labels_path, 'w') as f:
        f.write('\n'.join(all_labels))
    
    print(f"Created combined lists with {len(all_images)} images")
    
    return {
        'all_images': all_images_path,
        'all_labels': all_labels_path,
        'class_specific': {cls: os.path.join(output_dir, f"{cls}_images.txt") for cls in class_images if class_images[cls]}
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare ground truth images for CLIP embedding generation")
    parser.add_argument("--html", type=str, required=True, help="Path to HTML viewer file")
    parser.add_argument("--output_dir", type=str, default="gt_for_embeddings", help="Output directory for extracted images and lists")
    
    args = parser.parse_args()
    
    # Extract images from HTML
    image_paths, image_classes = extract_gt_images_from_html(args.html, args.output_dir)
    
    # Create class-specific lists
    output_files = create_class_specific_lists(image_paths, image_classes, args.output_dir)
    
    # Save output file paths
    output_json = os.path.join(args.output_dir, "gt_files.json")
    with open(output_json, 'w') as f:
        json.dump(output_files, f, indent=2)
    
    print(f"Ground truth preparation complete. Output files saved to {output_json}")

if __name__ == "__main__":
    main()
