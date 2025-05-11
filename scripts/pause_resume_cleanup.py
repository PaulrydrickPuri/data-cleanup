#!/usr/bin/env python3
"""
Pausable and resumable data cleanup pipeline for vehicle detection datasets.
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import time
import signal
import atexit

# Load environment variables
load_dotenv()

from data_cleanup.clean_dataset_single import clean_single_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pausable_cleanup.log")
    ]
)
logger = logging.getLogger(__name__)

class PausableCleanup:
    def __init__(self, dataset_path, output_dir, anchors_path, regex_path,
                 confidence=0.25, similarity=0.65, blur=100.0,
                 min_size=64, min_exposure=30, max_exposure=225,
                 log_raw_crops=False):
        """Initialize the pausable cleanup pipeline."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.anchors_path = Path(anchors_path)
        self.regex_path = Path(regex_path)
        
        self.confidence = confidence
        self.similarity = similarity
        self.blur = blur
        self.min_size = min_size
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.log_raw_crops = log_raw_crops
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # State file for tracking progress
        self.dataset_name = self.dataset_path.name
        self.state_file = self.output_dir / f"{self.dataset_name}_state.pkl"
        
        # Initialize or load state
        self.initialize_state()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        atexit.register(self.save_state)
        
        # Flag to indicate if processing should continue
        self.running = True
        
        # Initialize start time for progress tracking
        self.start_time = time.time()
    
    def initialize_state(self):
        """Initialize or load the processing state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    self.state = pickle.load(f)
                logger.info(f"Loaded existing state. {len(self.state['processed_images'])} images already processed.")
            except Exception as e:
                logger.error(f"Error loading state file: {e}")
                self.create_new_state()
        else:
            self.create_new_state()
    
    def create_new_state(self):
        """Create a new processing state."""
        self.state = {
            'processed_images': set(),
            'stats': {
                'total_images': 0,
                'processed_images': 0,
                'rejected_images': 0,
                'total_objects': 0,
                'accepted_objects': 0,
                'rejected_objects': 0,
                'rejection_reasons': {}
            },
            'last_processed_time': None
        }
    
    def save_state(self):
        """Save the current processing state."""
        self.state['last_processed_time'] = time.time()
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
        logger.info(f"State saved to {self.state_file}")
        
        # Also save stats as JSON for easier viewing
        stats_file = self.output_dir / f"{self.dataset_name}_stats.json"
        with open(stats_file, 'w') as f:
            # Create a copy of stats that's JSON serializable
            stats_copy = self.state['stats'].copy()
            stats_copy['processed_images_count'] = len(self.state['processed_images'])
            # Convert set to list for JSON serialization
            stats_copy['processed_images'] = list(self.state['processed_images'])
            json.dump(stats_copy, f, indent=2)
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals to pause processing."""
        logger.info("Received interrupt signal. Pausing processing...")
        self.running = False
    
    def find_images(self):
        """Find all images in the dataset."""
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    # Convert to absolute path for consistent tracking
                    image_path = os.path.abspath(image_path)
                    image_paths.append(image_path)
        return image_paths
    
    def format_time(self, seconds):
        """Format seconds into a human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def run(self):
        """Run the cleanup pipeline with pause/resume capability."""
        # Record start time for calculating processing rate
        self.start_time = time.time()
        
        # Find all images
        all_images = self.find_images()
        self.state['stats']['total_images'] = len(all_images)
        
        # Filter out already processed images
        images_to_process = [img for img in all_images if img not in self.state['processed_images']]
        
        logger.info(f"Found {len(all_images)} total images")
        logger.info(f"Already processed {len(self.state['processed_images'])} images")
        logger.info(f"Remaining {len(images_to_process)} images to process")
        
        if not images_to_process:
            logger.info("All images have been processed. Nothing to do.")
            return
        
        # Process images
        try:
            for i, image_path in enumerate(tqdm(images_to_process, desc="Processing images")):
                if not self.running:
                    logger.info("Processing paused. Progress saved.")
                    break
                
                try:
                    # Get relative path structure (use Path objects consistently)
                    image_path_obj = Path(image_path)
                    rel_path = os.path.relpath(str(image_path_obj), str(self.dataset_path))
                    img_dir = os.path.dirname(rel_path)
                    
                    # Sanitize output directory (handle special characters or spaces)
                    safe_dir = img_dir.replace(' ', '_').replace('(', '').replace(')', '')
                    
                    # Create corresponding directory in output
                    img_output_dir = Path(self.output_dir) / safe_dir
                    logger.info(f"Output directory: {img_output_dir} (type: {type(img_output_dir)})")
                    os.makedirs(img_output_dir, exist_ok=True)
                    
                    try:
                        # Process the image (single image at a time)
                        stats = clean_single_image(
                            image_path=Path(image_path),
                            output_dir=Path(img_output_dir),
                            anchors_path=self.anchors_path,
                            regex_path=self.regex_path,
                            confidence_threshold=self.confidence,
                            similarity_threshold=self.similarity,
                            blur_threshold=self.blur,
                            min_size=self.min_size,
                            min_exposure=self.min_exposure,
                            max_exposure=self.max_exposure,
                            log_raw_crops=self.log_raw_crops
                        )
                    except Exception as e:
                        logger.exception(f"Failed to process image {image_path}")
                        # Create minimal stats to track the error
                        stats = {
                            'total_images': 1,
                            'processed_images': 0,
                            'rejected_images': 1,
                            'total_objects': 0,
                            'accepted_objects': 0,
                            'rejected_objects': 0,
                            'rejection_reasons': {'processing_error': 1}
                        }
                    
                    # Update statistics
                    for key in stats:
                        if key == 'rejection_reasons':
                            for reason, count in stats['rejection_reasons'].items():
                                self.state['stats']['rejection_reasons'][reason] = self.state['stats']['rejection_reasons'].get(reason, 0) + count
                        else:
                            self.state['stats'][key] = self.state['stats'].get(key, 0) + stats.get(key, 0)
                    
                    # Mark as processed
                    self.state['processed_images'].add(image_path)
                    
                    # Calculate and log progress statistics
                    elapsed_time = time.time() - self.start_time
                    processed_count = len(self.state['processed_images'])
                    remaining_count = len(images_to_process) - (i + 1)
                    
                    # Only calculate time per image if we've processed at least one
                    if processed_count > 0:
                        time_per_image = elapsed_time / processed_count
                        estimated_remaining = time_per_image * remaining_count
                        
                        logger.info(f"Progress: {processed_count}/{len(all_images)} images ({processed_count/len(all_images)*100:.1f}%)")
                        logger.info(f"Time elapsed: {self.format_time(elapsed_time)}")
                        logger.info(f"Est. time remaining: {self.format_time(estimated_remaining)}")
                    
                    # Periodically save state
                    if i % 5 == 0:  
                        self.save_state()
                        
                        # Calculate elapsed time and estimate remaining time
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time
                        images_processed = len(self.state['processed_images'])
                        images_remaining = len(all_images) - images_processed
                        
                        # Calculate processing rate (images per second)
                        if elapsed_time > 0:
                            rate = images_processed / elapsed_time
                            estimated_time_remaining = images_remaining / rate if rate > 0 else 0
                            
                            # Format times for display
                            elapsed_time_str = self.format_time(elapsed_time)
                            remaining_time_str = self.format_time(estimated_time_remaining)
                            
                            # Print progress with time estimates
                            progress = images_processed / len(all_images) * 100
                            logger.info(f"Progress: {images_processed}/{len(all_images)} images ({progress:.1f}%)")
                            logger.info(f"Processing rate: {rate:.2f} images/second")
                            logger.info(f"Time elapsed: {elapsed_time_str}, estimated time remaining: {remaining_time_str}")
                        else:
                            # Print progress without time estimates if no time has elapsed
                            progress = images_processed / len(all_images) * 100
                            logger.info(f"Progress: {images_processed}/{len(all_images)} images ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # Still mark as processed to avoid getting stuck
                    self.state['processed_images'].add(image_path)
            
            # Final save
            self.save_state()
            
            # Print final statistics
            logger.info(f"Cleanup completed. Processed {len(self.state['processed_images'])}/{len(all_images)} images.")
            logger.info(f"Statistics:")
            logger.info(f"  Processed images: {self.state['stats']['processed_images']}")
            logger.info(f"  Rejected images: {self.state['stats']['rejected_images']}")
            logger.info(f"  Accepted objects: {self.state['stats']['accepted_objects']}")
            logger.info(f"  Rejected objects: {self.state['stats']['rejected_objects']}")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            self.save_state()

def main():
    parser = argparse.ArgumentParser(description="Run pausable data cleanup pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="cleaned_datasets", help="Base directory to save cleaned dataset")
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
    
    # Create cleaner and run
    cleaner = PausableCleanup(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
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
    
    cleaner.run()

if __name__ == "__main__":
    main()
