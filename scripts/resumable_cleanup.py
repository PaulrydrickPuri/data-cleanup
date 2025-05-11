#!/usr/bin/env python3
"""
Resumable version of the data cleanup pipeline for vehicle detection datasets.
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from data_cleanup.clean_dataset_single import clean_single_image

def run_resumable_cleanup(dataset_path, output_base_dir, anchors_path, regex_path, 
                          processed_file="processed_images.pkl",
                          confidence=0.25, similarity=0.65, blur=100.0,
                          min_size=64, min_exposure=30, max_exposure=225,
                          log_raw_crops=False):
    """
    Run the cleanup pipeline on a single dataset with resume capability.
    """
    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base_dir, dataset_name)
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Input path: {dataset_path}")
    print(f"Output path: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed images if exists
    processed_images_path = os.path.join(output_dir, processed_file)
    processed_images = set()
    
    if os.path.exists(processed_images_path):
        try:
            with open(processed_images_path, 'rb') as f:
                processed_images = pickle.load(f)
            print(f"Resuming from previous run. {len(processed_images)} images already processed.")
        except Exception as e:
            print(f"Error loading processed images file: {e}")
            processed_images = set()
    
    # Find all images in the dataset
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                # Convert to absolute path for consistent tracking
                image_path = os.path.abspath(image_path)
                image_paths.append(image_path)
    
    print(f"Found {len(image_paths)} images in dataset.")
    print(f"Skipping {len(processed_images)} already processed images.")
    
    # Filter out already processed images
    images_to_process = [img for img in image_paths if img not in processed_images]
    print(f"Processing {len(images_to_process)} remaining images.")
    
    if not images_to_process:
        print("All images have been processed. Nothing to do.")
        return
    
    # Process images in batches
    batch_size = 50  # Save progress every 50 images
    total_stats = {
        'total_images': 0,
        'processed_images': 0,
        'rejected_images': 0,
        'total_objects': 0,
        'accepted_objects': 0,
        'rejected_objects': 0,
        'rejection_reasons': {}
    }
    
    try:
        for i in range(0, len(images_to_process), batch_size):
            batch = images_to_process[i:i+batch_size]
            
            # Create a temporary directory for batch processing
            batch_dir = os.path.join(output_dir, f"batch_{i}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Process each image in the batch
            for img_path in tqdm(batch, desc=f"Batch {i//batch_size + 1}/{(len(images_to_process) + batch_size - 1) // batch_size}"):
                try:
                    # Get relative path structure
                    rel_path = os.path.relpath(img_path, dataset_path)
                    img_dir = os.path.dirname(rel_path)
                    
                    # Create corresponding directory in output
                    img_output_dir = os.path.join(output_dir, img_dir)
                    os.makedirs(img_output_dir, exist_ok=True)
                    
                    # Process individual image
                    img_stats = clean_single_image(
                        image_path=img_path,
                        output_dir=img_output_dir,
                        anchors_path=anchors_path,
                        regex_path=regex_path,
                        confidence_threshold=confidence,
                        similarity_threshold=similarity,
                        blur_threshold=blur,
                        min_size=min_size,
                        min_exposure=min_exposure,
                        max_exposure=max_exposure,
                        log_raw_crops=log_raw_crops
                    )
                    
                    # Update statistics
                    for key in ['total_images', 'processed_images', 'rejected_images', 
                                'total_objects', 'accepted_objects', 'rejected_objects']:
                        total_stats[key] += img_stats.get(key, 0)
                    
                    # Update rejection reasons
                    for reason, count in img_stats.get('rejection_reasons', {}).items():
                        total_stats['rejection_reasons'][reason] = total_stats['rejection_reasons'].get(reason, 0) + count
                    
                    # Mark as processed
                    processed_images.add(img_path)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Save progress after each batch
            with open(processed_images_path, 'wb') as f:
                pickle.dump(processed_images, f)
            
            # Save current stats
            stats_path = os.path.join(output_dir, "cleanup_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(total_stats, f, indent=2)
            
            print(f"\nProgress: {len(processed_images)}/{len(image_paths)} images processed ({len(processed_images)/len(image_paths)*100:.1f}%)")
            print(f"Cleaned dataset statistics so far:")
            print(f"  Total images: {total_stats['total_images']}")
            print(f"  Processed images: {total_stats['processed_images']}")
            print(f"  Rejected images: {total_stats['rejected_images']}")
            print(f"  Total objects: {total_stats['total_objects']}")
            print(f"  Accepted objects: {total_stats['accepted_objects']}")
            print(f"  Rejected objects: {total_stats['rejected_objects']}")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        print(f"Processed {len(processed_images)}/{len(image_paths)} images ({len(processed_images)/len(image_paths)*100:.1f}%)")
        
        # Save final stats
        stats_path = os.path.join(output_dir, "cleanup_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(total_stats, f, indent=2)
    
    # Final save of processed images
    with open(processed_images_path, 'wb') as f:
        pickle.dump(processed_images, f)
    
    print(f"\nAll done. Processed {len(processed_images)}/{len(image_paths)} images.")
    print(f"Final statistics saved to {os.path.join(output_dir, 'cleanup_stats.json')}")
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(description="Run resumable data cleanup pipeline on vehicle detection datasets")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Paths to dataset directories")
    parser.add_argument("--output_dir", default="cleaned_datasets", help="Base directory to save cleaned datasets")
    parser.add_argument("--anchors", type=str, default="data_cleanup/assets/gt_anchors_small.json", help="Path to anchor embeddings JSON file")
    parser.add_argument("--regex", type=str, default="data_cleanup/assets/regex_plate.json", help="Path to regex patterns JSON file")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for object detection")
    parser.add_argument("--similarity", type=float, default=0.65, help="Similarity threshold for class validation")
    parser.add_argument("--blur", type=float, default=100.0, help="Blur threshold")
    parser.add_argument("--min_size", type=int, default=64, help="Minimum size")
    parser.add_argument("--min_exposure", type=int, default=30, help="Minimum exposure")
    parser.add_argument("--max_exposure", type=int, default=225, help="Maximum exposure")
    parser.add_argument("--log_raw_crops", action="store_true", help="Save rejected crops for inspection")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of images to process before saving progress")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    all_stats = {}
    
    for dataset_path in args.datasets:
        try:
            stats = run_resumable_cleanup(
                dataset_path=dataset_path,
                output_base_dir=args.output_dir,
                anchors_path=args.anchors,
                regex_path=args.regex,
                confidence=args.confidence,
                similarity=args.similarity,
                blur=args.blur,
                min_size=args.min_size,
                min_exposure=args.min_exposure,
                max_exposure=args.max_exposure,
                log_raw_crops=args.log_raw_crops
            )
            
            dataset_name = os.path.basename(dataset_path)
            all_stats[dataset_name] = stats
            
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {e}")
    
    # Save overall stats
    overall_stats_path = os.path.join(args.output_dir, "overall_stats.json")
    with open(overall_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nAll datasets processed. Overall stats saved to {overall_stats_path}")

if __name__ == "__main__":
    main()
