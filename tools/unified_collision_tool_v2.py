#!/usr/bin/env python3
"""
Unified Collision Detection Tool (v2)

This script provides a single entry point for running different collision detection algorithms
organized into intuitive categories based on complexity and use case.

Categories:
1. DETECTION METHODS (Basic Tier)
   - standard: Basic IoU-based collision detection
   - safety: Focused on pedestrian/vehicle interactions
   - quick: Rapid validation of detection pipeline

2. ADVANCED ANALYTICS (Advanced Tier)
   - enhanced: Higher accuracy collision detection with multi-model approach
   - centroid: Proximity-based detection with heatmap generation
   - tracking: Advanced object tracking across frames

3. BATCH PROCESSING (Admin Tier)
   - batch: Process multiple videos in sequence
   - calibration: Tune detection parameters and validate against ground truth
   - telemetry: Collect performance metrics and log processing statistics

Detailed Method Parameters:
--------------------------

STANDARD DETECTION
* conf_threshold: Confidence threshold for object detection (default: 0.25)
* iou_threshold: IoU threshold for collision detection (default: 0.2)
* Suitable for: General purpose videos with clear object separation
* Output: Annotated video with bounding boxes and collision markers

SAFETY ANALYSIS
* conf_threshold: Confidence threshold for object detection (default: 0.25)
* danger_threshold: Pixel distance threshold for proximity alerts (default: 150)
* Suitable for: Traffic monitoring, pedestrian safety analysis
* Output: Safety-annotated video with proximity warnings

QUICK TEST
* conf_threshold: Confidence threshold for object detection (default: 0.3)
* Suitable for: Rapid validation of detection pipeline
* Output: Basic annotated video with minimal processing

ENHANCED DETECTION
* conf_threshold: Confidence threshold for object detection (default: 0.3)
* iou_threshold: IoU threshold for collision detection (default: 0.25)
* Suitable for: Complex scenes requiring higher accuracy
* Output: High-quality annotated video with detailed collision analysis

CENTROID ANALYTICS
* conf_threshold: Confidence threshold for object detection (default: 0.3)
* proximity_threshold: Pixel distance for proximity detection (default: 80)
* Suitable for: Crowded scenes where bounding boxes overlap
* Output: Video with centroid tracking + heatmap of collision hotspots

TRACKING ANALYSIS
* conf_threshold: Confidence threshold for object detection (default: 0.35)
* iou_threshold: IoU threshold for tracking continuity (default: 0.2)
* Suitable for: Videos with object occlusion or camera movement
* Output: Video with persistent object tracking and trajectory analysis

BATCH ANALYSIS
* conf_threshold: Confidence threshold for object detection (default: 0.25)
* iou_threshold: IoU threshold for collision detection (default: 0.2)
* export_format: Format for results export (json, csv, both)
* Suitable for: Processing multiple videos with consistent parameters
* Output: Multiple annotated videos + aggregated results file

CALIBRATION ANALYSIS
* conf_threshold: Starting confidence threshold (default: 0.3)
* iou_threshold: Starting IoU threshold (default: 0.2)
* Suitable for: Optimizing detection parameters against ground truth
* Output: Parameter sweep results with precision/recall metrics

TELEMETRY ANALYSIS
* conf_threshold: Confidence threshold for object detection (default: 0.25)
* Suitable for: Performance benchmarking and system monitoring
* Output: Detailed performance metrics and processing statistics
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to allow importing modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Default paths and thresholds
DEFAULT_MODELS = {
    "general": "yolov8n.pt",  # COCO model for general detection
    "vehicle": "models/outputs/vehicle_detection2/weights/best.pt",  # Specialized vehicle model
    "person": "models/outputs/person_detection/weights/best.pt"  # Specialized person model
}

DEFAULT_THRESHOLDS = {
    "standard": {"conf": 0.25, "iou": 0.2},
    "safety": {"conf": 0.25, "danger": 150},
    "quick": {"conf": 0.3, "iou": 0.15},
    "enhanced": {"conf": 0.3, "iou": 0.25},
    "centroid": {"conf": 0.3, "proximity": 80},
    "tracking": {"conf": 0.35, "iou": 0.2},
    "batch": {"conf": 0.25, "iou": 0.2},
    "calibration": {"conf": 0.3, "iou": 0.2},
    "telemetry": {"conf": 0.25, "iou": 0.2}
}

# Method categories
METHOD_CATEGORIES = {
    "basic": ["standard", "safety", "quick"],
    "advanced": ["enhanced", "centroid", "tracking"],
    "admin": ["batch", "calibration", "telemetry"]
}

# Flattened methods list
ALL_METHODS = [method for category in METHOD_CATEGORIES.values() for method in category]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified tool for collision detection analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard detection on a video
  python unified_collision_tool_v2.py --method standard path/to/video.mp4
  
  # Run enhanced detection with custom threshold
  python unified_collision_tool_v2.py --method enhanced --threshold 0.35 path/to/video.mp4
  
  # Run batch processing on a directory of videos
  python unified_collision_tool_v2.py --method batch --input-dir path/to/videos/
        """
    )
    
    parser.add_argument(
        "input",
        help="Path to input video file or directory (for batch processing)"
    )
    
    parser.add_argument(
        "--method",
        choices=ALL_METHODS,
        default="standard",
        help="""Detection method to use:
BASIC TIER:
  standard: Basic IoU-based collision detection
  safety: Focused on pedestrian/vehicle interactions
  quick: Rapid validation of detection pipeline

ADVANCED TIER:
  enhanced: Higher accuracy collision detection
  centroid: Proximity-based detection with heatmaps
  tracking: Advanced object tracking across frames

ADMIN TIER:
  batch: Process multiple videos in sequence
  calibration: Tune and validate detection parameters
  telemetry: Collect performance metrics"""
    )
    
    parser.add_argument(
        "--output",
        help="Path to save the output video (default: auto-generated based on input)",
        default=None
    )
    
    parser.add_argument(
        "--general-model",
        help=f"Path to the general detection model (default: {DEFAULT_MODELS['general']})",
        default=DEFAULT_MODELS["general"]
    )
    
    parser.add_argument(
        "--vehicle-model",
        help=f"Path to the vehicle detection model (default: {DEFAULT_MODELS['vehicle']})",
        default=DEFAULT_MODELS["vehicle"]
    )
    
    parser.add_argument(
        "--person-model",
        help=f"Path to the person detection model (default: {DEFAULT_MODELS['person']})",
        default=DEFAULT_MODELS["person"]
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Detection confidence threshold (default: method-specific)",
        default=None
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        help="IoU threshold for collision detection (default: method-specific)",
        default=None
    )
    
    parser.add_argument(
        "--input-dir",
        help="Directory containing videos for batch processing",
        default=None
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Export format for results (default: json)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of results"
    )
    
    parser.add_argument(
        "--category",
        choices=list(METHOD_CATEGORIES.keys()),
        help="List all methods in a specific category",
        default=None
    )
    
    return parser.parse_args()

def list_methods_in_category(category):
    """List all methods in a specific category"""
    if category not in METHOD_CATEGORIES:
        logger.error(f"Unknown category: {category}")
        return
    
    print(f"\n{category.upper()} TIER METHODS:")
    for method in METHOD_CATEGORIES[category]:
        print(f"  - {method}")
        # Get the docstring from the function if available
        func_name = f"run_{method}_analysis"
        if func_name in globals():
            doc = globals()[func_name].__doc__
            if doc:
                print(f"    {doc.strip().split('\n')[0]}")
    print()

def get_default_output_path(input_path, method):
    """Generate a default output path based on the input path and method"""
    input_path = Path(input_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs") / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    timestamp = int(time.time())
    if input_path.is_dir():
        return output_dir / f"{method}_batch_{timestamp}"
    else:
        return output_dir / f"{input_path.stem}_{method}_{timestamp}.mp4"

#
# BASIC TIER METHODS
#

def run_standard_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run standard collision detection using IoU-based approach
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for person detection
        - vehicle: Specialized vehicle detection model
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.25)
        - iou: IoU threshold for collision detection (default: 0.2)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with collision annotations
    
    Notes:
    ------
    This method uses the basic collision_detection module to detect collisions
    between people and vehicles using Intersection over Union (IoU) calculations.
    It's suitable for general-purpose videos with clear object separation.
    """
    logger.info("Running standard collision detection")
    
    # Import collision detection module
    try:
        from tools.collision_detection import detect_collisions
    except ImportError:
        try:
            from collision_detection import detect_collisions
        except ImportError:
            logger.error("Could not import collision_detection module")
            return None
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["standard"]["conf"])
    iou_threshold = thresholds.get("iou", DEFAULT_THRESHOLDS["standard"]["iou"])
    
    # Run detection
    try:
        output_file = detect_collisions(
            video_path=input_path,
            general_model_path=models["general"],
            vehicle_model_path=models["vehicle"],
            output_path=str(output_path),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        logger.info(f"Standard analysis complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in standard analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_safety_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run safety analysis focused on pedestrian/vehicle interactions
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for person detection
        - vehicle: Specialized vehicle detection model
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.25)
        - danger: Pixel distance threshold for proximity alerts (default: 150)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with safety annotations
    
    Notes:
    ------
    This method focuses specifically on pedestrian safety around vehicles,
    using pixel distance measurements rather than IoU. It generates proximity
    warnings when pedestrians and vehicles come within the danger threshold.
    Particularly useful for traffic monitoring and pedestrian safety analysis.
    """
    logger.info("Running safety analysis for pedestrian/vehicle interactions")
    
    # Import safety analysis module
    try:
        from tools.run_safety_analysis import run_safety_analysis_main
    except ImportError:
        try:
            from safety_detection import run_safety_analysis_main
        except ImportError:
            logger.warning("Safety analysis module not available. Falling back to standard detection.")
            return run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["safety"]["conf"])
    danger_threshold = thresholds.get("danger", DEFAULT_THRESHOLDS["safety"]["danger"])
    
    # Run analysis
    try:
        output_file = run_safety_analysis_main(
            video_path=input_path,
            general_model_path=models["general"],
            vehicle_model_path=models["vehicle"],
            output_path=str(output_path),
            conf_threshold=conf_threshold,
            danger_threshold=danger_threshold
        )
        
        logger.info(f"Safety analysis complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in safety analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_quick_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run quick test analysis for rapid validation
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model (only this model is used)
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.3)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with basic annotations
    
    Notes:
    ------
    This method is designed for quick validation of the detection pipeline.
    It uses simplified processing with only the general detection model and
    minimal post-processing. It's ideal for quickly checking if the system
    is working correctly or for initial testing of new videos.
    """
    logger.info("Running quick test analysis")
    
    # Import quick test module
    try:
        from tools.run_quick_smoke_test import run_quick_test
    except ImportError:
        try:
            from tools.run_smoke_test import run_smoke_test as run_quick_test
        except ImportError:
            logger.warning("Quick test module not available. Falling back to standard detection with lower thresholds.")
            # Use standard detection with lower thresholds for speed
            quick_thresholds = {
                "conf": 0.3,
                "iou": 0.15
            }
            return run_standard_analysis(input_path, output_path, models, quick_thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["quick"]["conf"])
    
    # Run quick test
    try:
        output_file = run_quick_test(
            video_path=input_path,
            model_path=models["general"],
            output_path=str(output_path),
            conf_threshold=conf_threshold
        )
        
        logger.info(f"Quick test complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in quick test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

#
# ADVANCED TIER METHODS
#

def run_enhanced_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run enhanced collision detection with multi-model approach
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for person detection
        - vehicle: Specialized vehicle detection model
        - person: Specialized person detection model (optional)
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.3)
        - iou: IoU threshold for collision detection (default: 0.25)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with detailed collision annotations
    
    Notes:
    ------
    This method uses an enhanced collision detection algorithm with a multi-model
    approach. It combines specialized models for better accuracy and includes
    advanced tracking to maintain object identity across frames. It's suitable
    for complex scenes with multiple objects and potential occlusions.
    
    This method is more computationally intensive than standard detection but
    provides higher accuracy and more detailed analysis.
    """
    logger.info("Running enhanced collision detection")
    
    # Import enhanced detection module
    try:
        from tools.enhanced_collision_detection import detect_collisions_enhanced
    except ImportError:
        try:
            from enhanced_collision_detection import detect_collisions_enhanced
        except ImportError:
            logger.warning("Enhanced detection module not available. Falling back to standard detection.")
            return run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["enhanced"]["conf"])
    iou_threshold = thresholds.get("iou", DEFAULT_THRESHOLDS["enhanced"]["iou"])
    
    # Run enhanced detection
    try:
        output_file = detect_collisions_enhanced(
            video_path=input_path,
            general_model_path=models["general"],
            vehicle_model_path=models["vehicle"],
            output_path=str(output_path),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        logger.info(f"Enhanced analysis complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_centroid_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run centroid-based detection with heatmap generation
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model (primary model used)
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.3)
        - proximity: Pixel distance threshold for proximity detection (default: 80)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with centroid tracking and heatmap visualization
    
    Notes:
    ------
    This method uses a centroid-based approach for collision detection instead of IoU.
    It tracks the center points of objects and measures the distance between them,
    which works better in crowded scenes where bounding boxes frequently overlap.
    
    The method also generates heatmaps showing collision hotspots over time,
    which is useful for identifying problematic areas in the scene. Results are
    saved to a dedicated centroid_analytics_results directory with timestamped segments.
    """
    logger.info("Running centroid-based detection with heatmap")
    
    # Import centroid detection module
    try:
        from tools.run_centroid_detection import run_centroid_detection
    except ImportError:
        try:
            from run_centroid_analytics import run_centroid_analytics as run_centroid_detection
        except ImportError:
            logger.warning("Centroid detection module not available. Falling back to standard detection.")
            return run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["centroid"]["conf"])
    proximity_threshold = thresholds.get("proximity", DEFAULT_THRESHOLDS["centroid"]["proximity"])
    
    # Create timestamp-based output path if needed
    timestamp = int(time.time())
    if output_path is None:
        output_dir = Path("centroid_analytics_results") / f"segment_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"centroid_segment_{timestamp}_output_viz.mp4"
    
    # Run centroid detection
    try:
        output_file = run_centroid_detection(
            video_path=input_path,
            output_path=str(output_path),
            model_path=models["general"],
            conf_threshold=conf_threshold,
            proximity_threshold=proximity_threshold
        )
        
        logger.info(f"Centroid analysis complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in centroid analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_tracking_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run advanced object tracking across frames
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for object detection
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.35)
        - iou: IoU threshold for tracking continuity (default: 0.2)
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    str
        Path to the output video file with persistent tracking visualization
    
    Notes:
    ------
    This method focuses on advanced object tracking across frames using algorithms
    like StrongSORT, Kalman filters, or sliding threshold tracking. It maintains
    object identity even through occlusions or when objects temporarily leave the frame.
    
    The tracking analysis is particularly useful for videos with complex object
    movements, camera motion, or frequent occlusions. It generates trajectory
    analysis and can identify near-misses based on projected paths rather than
    just current positions.
    """
    logger.info("Running advanced object tracking analysis")
    
    # Import tracking module
    try:
        from tools.run_persistent_tracking import run_persistent_tracking
    except ImportError:
        try:
            from tools.run_sliding_threshold_tracking import run_sliding_threshold_tracking as run_persistent_tracking
        except ImportError:
            logger.warning("Tracking module not available. Falling back to standard detection.")
            return run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["tracking"]["conf"])
    iou_threshold = thresholds.get("iou", DEFAULT_THRESHOLDS["tracking"]["iou"])
    
    # Run tracking analysis
    try:
        output_file = run_persistent_tracking(
            video_path=input_path,
            output_path=str(output_path),
            model_path=models["general"],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        logger.info(f"Tracking analysis complete. Output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in tracking analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

#
# ADMIN TIER METHODS
#

def run_batch_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run batch processing on multiple videos
    
    Parameters:
    -----------
    input_path : str
        Path to the input directory containing videos to process
        (If a single file is provided, input_dir from kwargs will be used instead)
    output_path : str
        Path to save the output videos and aggregated results
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for person detection
        - vehicle: Specialized vehicle detection model
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.25)
        - iou: IoU threshold for collision detection (default: 0.2)
    **kwargs : dict
        Additional keyword arguments:
        - input_dir: Alternative directory for batch processing
        - export_format: Format for results export (json, csv, both)
        - visualize: Whether to generate visualization videos
        
    Returns:
    --------
    str
        Path to the batch results file (JSON or CSV)
    
    Notes:
    ------
    This method processes multiple videos in sequence using the same parameters
    and models. It generates individual output videos for each input and an
    aggregated results file with collision statistics across all videos.
    
    Batch processing is useful for analyzing large datasets, such as traffic
    camera footage over multiple days or different camera angles of the same scene.
    The aggregated results allow for trend analysis and pattern identification.
    """
    logger.info("Running batch analysis on multiple videos")
    
    # Check if input is a directory
    input_path = Path(input_path)
    if not input_path.is_dir():
        if kwargs.get("input_dir"):
            input_path = Path(kwargs["input_dir"])
        else:
            logger.error("Batch processing requires a directory. Use --input-dir or provide a directory as input.")
            return None
    
    # Import batch processing module
    try:
        from tools.run_batch_analytics import run_batch_analytics
    except ImportError:
        logger.warning("Batch processing module not available. Processing videos individually.")
        
        # Process each video individually
        results = []
        for video_file in input_path.glob("*.mp4"):
            logger.info(f"Processing video: {video_file}")
            video_output = output_path / f"{video_file.stem}_output.mp4"
            result = run_standard_analysis(str(video_file), str(video_output), models, thresholds)
            if result:
                results.append({"video": str(video_file), "output": result})
        
        # Save batch results
        batch_results_file = output_path / "batch_results.json"
        with open(batch_results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch processing complete. Results saved to: {batch_results_file}")
        return batch_results_file
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["batch"]["conf"])
    iou_threshold = thresholds.get("iou", DEFAULT_THRESHOLDS["batch"]["iou"])
    
    # Run batch processing
    try:
        output_file = run_batch_analytics(
            input_dir=str(input_path),
            output_dir=str(output_path),
            general_model_path=models["general"],
            vehicle_model_path=models["vehicle"],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            export_format=kwargs.get("export_format", "json")
        )
        
        logger.info(f"Batch analysis complete. Results saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_calibration_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run calibration and validation against ground truth
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file or directory with ground truth annotations
    output_path : str
        Path to save the calibration results and parameter sweep data
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model for person detection
        - vehicle: Specialized vehicle detection model
    thresholds : dict
        Dictionary containing starting detection thresholds:
        - conf: Starting confidence threshold for sweep (default: 0.3)
        - iou: Starting IoU threshold for sweep (default: 0.2)
    **kwargs : dict
        Additional keyword arguments:
        - sweep_range: Range for parameter sweep (default: 0.05-0.95)
        - sweep_steps: Number of steps in parameter sweep (default: 10)
        - metrics: List of metrics to calculate (precision, recall, f1)
        
    Returns:
    --------
    str
        Path to the calibration results file with optimal parameters
    
    Notes:
    ------
    This method performs parameter calibration by running detection with various
    threshold combinations and comparing the results against ground truth annotations.
    It generates precision-recall curves and determines the optimal parameter
    settings for the specific dataset.
    
    Calibration is particularly useful when deploying the system in a new environment
    or with a new camera setup, as it helps tune the parameters for optimal performance.
    The results include recommended thresholds for production use.
    """
    logger.info("Running calibration and validation")
    
    # Import calibration module
    try:
        from tools.run_calibration_v2_validation import run_calibration_validation
    except ImportError:
        try:
            from tools.run_calibration_validation import run_calibration_validation
        except ImportError:
            logger.warning("Calibration module not available. Falling back to standard detection.")
            return run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["calibration"]["conf"])
    iou_threshold = thresholds.get("iou", DEFAULT_THRESHOLDS["calibration"]["iou"])
    
    # Run calibration
    try:
        output_file = run_calibration_validation(
            video_path=input_path,
            output_dir=str(output_path),
            general_model_path=models["general"],
            vehicle_model_path=models["vehicle"],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        logger.info(f"Calibration complete. Results saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in calibration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_telemetry_analysis(input_path, output_path, models, thresholds, **kwargs):
    """
    Run telemetry collection and performance metrics
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file for telemetry collection
    output_path : str
        Path to save the telemetry results and performance metrics
    models : dict
        Dictionary containing paths to detection models:
        - general: General COCO-trained model (primary model used)
    thresholds : dict
        Dictionary containing detection thresholds:
        - conf: Confidence threshold for object detection (default: 0.25)
    **kwargs : dict
        Additional keyword arguments:
        - metrics_format: Format for metrics export (json, csv, loki)
        - collect_gpu: Whether to collect GPU metrics (default: True)
        - collect_memory: Whether to collect memory usage (default: True)
        - collect_timing: Whether to collect timing information (default: True)
        
    Returns:
    --------
    str
        Path to the telemetry results file with performance metrics
    
    Notes:
    ------
    This method focuses on collecting performance metrics and telemetry data
    during the execution of the collision detection pipeline. It measures
    processing time, memory usage, GPU utilization, and other system metrics
    to help optimize the system for production use.
    
    Telemetry collection is particularly useful for benchmarking different
    hardware configurations, optimizing resource allocation, and identifying
    performance bottlenecks. The results can be exported to various formats
    including JSON, CSV, or sent to a Loki logging system for monitoring.
    """
    logger.info("Running telemetry and performance metrics collection")
    
    # Import telemetry module
    try:
        from tools.run_telemetry_experiment import run_telemetry_experiment
    except ImportError:
        try:
            from tools.metrics_exporter import export_metrics
        except ImportError:
            logger.warning("Telemetry module not available. Falling back to standard detection with metrics.")
            # Run standard detection and collect basic metrics
            start_time = time.time()
            result = run_standard_analysis(input_path, output_path, models, thresholds, **kwargs)
            duration = time.time() - start_time
            
            # Save basic metrics
            metrics = {
                "video": str(input_path),
                "method": "standard",
                "duration_seconds": duration,
                "output": str(result) if result else None,
                "timestamp": time.time()
            }
            
            metrics_file = Path(output_path).parent / f"{Path(output_path).stem}_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Basic metrics saved to: {metrics_file}")
            return metrics_file
    
    # Set default thresholds if not provided
    conf_threshold = thresholds.get("conf", DEFAULT_THRESHOLDS["telemetry"]["conf"])
    
    # Run telemetry collection
    try:
        output_file = run_telemetry_experiment(
            video_path=input_path,
            output_dir=str(output_path),
            model_path=models["general"],
            conf_threshold=conf_threshold
        )
        
        logger.info(f"Telemetry collection complete. Results saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in telemetry collection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to run the unified tool"""
    args = parse_arguments()
    
    # If category is specified, just list methods in that category and exit
    if args.category:
        list_methods_in_category(args.category)
        return
    
    # Validate input path
    input_path = args.input
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Set method
    method = args.method
    
    # Determine which category the method belongs to
    method_category = None
    for category, methods in METHOD_CATEGORIES.items():
        if method in methods:
            method_category = category
            break
    
    if method_category is None:
        logger.error(f"Unknown method: {method}")
        sys.exit(1)
    
    logger.info(f"Running {method} analysis (Category: {method_category})")
    
    # Set output path if not provided
    if args.output is None:
        output_path = get_default_output_path(input_path, method)
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare models dictionary
    models = {
        "general": args.general_model,
        "vehicle": args.vehicle_model,
        "person": args.person_model
    }
    
    # Prepare thresholds dictionary
    thresholds = {}
    if args.threshold is not None:
        thresholds["conf"] = args.threshold
    if args.iou_threshold is not None:
        thresholds["iou"] = args.iou_threshold
    
    # Prepare additional kwargs
    kwargs = {
        "input_dir": args.input_dir,
        "export_format": args.export_format,
        "visualize": args.visualize
    }
    
    # Run the selected method
    start_time = time.time()
    
    try:
        # Map method to function
        method_func_map = {
            "standard": run_standard_analysis,
            "safety": run_safety_analysis,
            "quick": run_quick_analysis,
            "enhanced": run_enhanced_analysis,
            "centroid": run_centroid_analysis,
            "tracking": run_tracking_analysis,
            "batch": run_batch_analysis,
            "calibration": run_calibration_analysis,
            "telemetry": run_telemetry_analysis
        }
        
        if method not in method_func_map:
            logger.error(f"Method {method} is not implemented yet")
            sys.exit(1)
        
        # Call the appropriate function
        output_file = method_func_map[method](input_path, output_path, models, thresholds, **kwargs)
        
        if output_file:
            duration = time.time() - start_time
            logger.info(f"Processing completed in {duration:.2f} seconds")
            logger.info(f"Output saved to: {output_file}")
        else:
            logger.error(f"Processing failed for method: {method}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error running {method} analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
