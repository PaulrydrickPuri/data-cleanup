#!/usr/bin/env python3
"""
Script to run the data cleanup pipeline on vehicle detection datasets.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import logging
from tqdm import tqdm

from data_cleanup.clean_dataset import clean_dataset

def run_cleanup_on_dataset(dataset_path, output_base_dir, anchors_path, regex_path, 
                          confidence=0.25, similarity=0.65, blur=100.0,
                          min_size=64, min_exposure=30, max_exposure=225,
                          log_raw_crops=False):
    """
    Run the cleanup pipeline on a single dataset.
    """
    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base_dir, dataset_name)
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Input path: {dataset_path}")
    print(f"Output path: {output_dir}")
    
    # Run the cleanup pipeline
    stats = clean_dataset(
        input_dir=dataset_path,
        output_dir=output_dir,
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
    
    # Save stats to a JSON file
    stats_path = os.path.join(output_dir, "cleanup_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\nCleaned dataset statistics for {dataset_name}:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Processed images: {stats['processed_images']}")
    print(f"  Rejected images: {stats['rejected_images']}")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Accepted objects: {stats['accepted_objects']}")
    print(f"  Rejected objects: {stats['rejected_objects']}")
    print(f"  Rejection reasons:")
    for reason, count in stats['rejection_reasons'].items():
        print(f"    {reason}: {count}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Run data cleanup pipeline on vehicle detection datasets")
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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    all_stats = {}
    
    for dataset_path in args.datasets:
        try:
            stats = run_cleanup_on_dataset(
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
