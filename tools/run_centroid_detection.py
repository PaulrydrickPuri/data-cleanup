#!/usr/bin/env python3
"""
Run the optimized centroid-based collision detection on a video file
and generate heat-map analytics of collision hotspots.

This implementation includes:
1. Centroid-based detection (80px proximity threshold)
2. 2-second yellow flag cooldown to reduce alert fatigue
3. DBSCAN clustering for hotspot identification
4. Heat-map visualization of near-miss patterns
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
from datetime import datetime

# Import collision detection modules
from ultralytics import YOLO
from collision_detection import calculate_iou, ObjectTracker
from collision_config import *
from tracking_persistence import TrackingSystem
from cooldown_manager import CooldownManager
from heatmap_analytics import HeatmapAnalyzer, NearMissEvent
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('centroid_detection')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Run centroid-based collision detection and generate heat-maps")
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--output", type=str, default=None, help="Path to output video file")
parser.add_argument("--heatmap-only", action="store_true", help="Only generate heat-map, no video output")
parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0') or 'cpu'")
args = parser.parse_args()

def main():
    """Main function to run optimized collision detection and generate heat-maps."""
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Initialize video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return 1

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output path if not specified
    if args.output is None:
        base_dir = os.path.dirname(args.video)
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(base_dir, f"{base_name}_centroid_detection.mp4")
    
    # Set up video writer if not heat-map only mode
    if not args.heatmap_only:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize tracking system
    tracking_system = TrackingSystem()
    
    # Load models
    try:
        # Load YOLOv8 model for detection
        model = YOLO('yolov8n.pt')
        logger.info(f"Loaded YOLOv8 model for detection")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Compute resolution-appropriate centroid threshold
    dynamic_centroid_threshold = compute_centroid_threshold(width, height)
    logger.info(f"Video resolution: {width}x{height}")
    logger.info(f"Using dynamic centroid threshold: {dynamic_centroid_threshold:.1f}px")
    
    # Initialize cooldown manager for yellow flags
    cooldown_manager = CooldownManager(cooldown_time=YELLOW_COOLDOWN_TIME)
    
    # Initialize heat-map analyzer with dynamic threshold
    frame_shape = (height, width, 3)
    heatmap_analyzer = HeatmapAnalyzer(
        frame_shape=frame_shape,
        centroid_threshold=dynamic_centroid_threshold
    )
    
    # Initialize trackers for people and vehicles
    person_trackers = {}
    vehicle_trackers = {}
    last_id = 0
    
    # Statistics
    start_time = time.time()
    frame_count = 0
    collision_count = 0
    potential_count = 0
    yellow_per_minute = 0
    processing_times = []
    
    # For CI thresholds
    CI_MAX_YELLOW_PER_MINUTE = 30.0  # Fail if exceeded
    CI_MIN_COLLISIONS = 5            # Minimum expected collisions
    CI_MAX_FPS_DROP_PCT = 20.0       # Maximum FPS degradation %
    
    # Process each frame
    logger.info(f"Processing video with {total_frames} frames...")
    logger.info(f"Using centroid threshold: {dynamic_centroid_threshold:.1f}px with {YELLOW_COOLDOWN_TIME}s cooldown")
    
    # Save a frame for background
    background_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store first clean frame as background for heat-map
        if background_frame is None and frame is not None:
            background_frame = frame.copy()
        
        # Process frame for object detection
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_idx / fps
        
        # Run YOLO detection
        results = model(frame, conf=args.confidence, iou=args.iou, classes=[0, 2, 3, 5, 7])
        detections = results[0].boxes
        
        # Process detections
        persons = []
        vehicles = []
        
        for det in detections:
            # Get detection info
            box = det.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            cls = int(det.cls[0].item())     # class ID
            conf = det.conf[0].item()        # confidence
            
            # Create object format
            obj = {'class': cls, 'box': box, 'confidence': conf}
            
            # Apply tracking using tracking_system
            track_id = tracking_system.update_object_tracking(cls, box, frame_idx)
            obj['track_id'] = track_id
            
            # Categorize objects
            if cls == 0:  # person
                persons.append(obj)
            elif cls in [2, 3, 5, 7]:  # vehicle classes
                vehicles.append(obj)
        
        # Process each person-vehicle pair for collision detection
        frame_has_collision = False
        frame_has_potential = False
        
        for person in persons:
            p_id = person['track_id']
            p_box = person['box']
            p_center = ((p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2)
            
            for vehicle in vehicles:
                v_id = vehicle['track_id']
                v_box = vehicle['box']
                v_center = ((v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2)
                
                # Calculate IoU - box format in calculate_iou is [x1, y1, w, h]
                p_rect = [p_box[0], p_box[1], p_box[2] - p_box[0], p_box[3] - p_box[1]]
                v_rect = [v_box[0], v_box[1], v_box[2] - v_box[0], v_box[3] - v_box[1]]
                iou = calculate_iou(p_rect, v_rect)
                
                # Calculate centroid distance for detection
                centroid_distance = np.hypot(p_center[0] - v_center[0], p_center[1] - v_center[1])
                
                # Get velocity change from tracking system if available
                velocity_change = tracking_system.get_velocity_change(p_id) if p_id in tracking_system.trackers else 0.0
                
                # Update EMA values
                interaction_id = f"p{p_id}_v{v_id}"
                smoothed_iou, smoothed_vel = update_ema_values(interaction_id, iou, velocity_change)
                
                # Get collision state with centroid detection - use dynamic threshold
                is_collision, is_potential = get_collision_state(
                    iou, velocity_change, 
                    smoothed_iou, smoothed_vel,
                    p_center, v_center,
                    cooldown_manager, current_time,
                    None, frame_idx, p_id, v_id
                )
                
                # For regression testing, log diagnostics on important decisions
                if is_collision or is_potential or (centroid_distance < dynamic_centroid_threshold * 1.2):
                    logger.debug(
                        f"Frame {frame_idx}: P{p_id}-V{v_id} d={centroid_distance:.1f}px "
                        f"IoU={iou:.3f} EMA_IoU={smoothed_iou:.3f} Detection={'RED' if is_collision else 'YELLOW' if is_potential else 'NONE'}"
                    )
                
                # Record event if it's a collision or potential collision
                if is_collision or is_potential:
                    # Calculate the actual centroid distance for the event
                    centroid_distance = np.hypot(p_center[0] - v_center[0], p_center[1] - v_center[1])
                    
                    # Create and add event to analyzer
                    event = NearMissEvent(
                        frame_idx=frame_idx,
                        timestamp=current_time,
                        p_center=p_center,
                        v_center=v_center,
                        iou=iou,
                        centroid_distance=centroid_distance,
                        is_collision=is_collision
                    )
                    heatmap_analyzer.add_event(event)
                
                # Update frame collision flags
                if is_collision:
                    frame_has_collision = True
                    collision_count += 1
                    
                    # Draw collision visualization
                    cv2.rectangle(frame, (int(p_box[0]), int(p_box[1])), 
                                 (int(p_box[2]), int(p_box[3])), (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(v_box[0]), int(v_box[1])), 
                                 (int(v_box[2]), int(v_box[3])), (0, 0, 255), 2)
                    
                    # Draw line between centroids
                    cv2.line(frame, 
                           (int(p_center[0]), int(p_center[1])), 
                           (int(v_center[0]), int(v_center[1])),
                           (0, 0, 255), 2)
                    
                elif is_potential:
                    frame_has_potential = True
                    potential_count += 1
                    
                    # Draw potential collision visualization
                    cv2.rectangle(frame, (int(p_box[0]), int(p_box[1])), 
                                 (int(p_box[2]), int(p_box[3])), (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(v_box[0]), int(v_box[1])), 
                                 (int(v_box[2]), int(v_box[3])), (0, 255, 255), 2)
                    
                    # Draw line between centroids
                    cv2.line(frame, 
                           (int(p_center[0]), int(p_center[1])), 
                           (int(v_center[0]), int(v_center[1])),
                           (0, 255, 255), 1)
        
        # Add HUD information
        minutes_elapsed = max(1, frame_count / (fps * 60))
        yellow_per_minute = potential_count / minutes_elapsed
        
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Collisions: {collision_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Potentials: {potential_count} ({yellow_per_minute:.1f}/min)", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Apply background visual effects for alerts
        if frame_has_collision:
            # Red vignette for collision
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 128), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "COLLISION DETECTED", (width//2 - 150, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif frame_has_potential:
            # Yellow vignette for potential
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 128, 128), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Calculate processing metrics
        processing_end = time.time()
        frame_processing_time = processing_end - (start_time + sum(processing_times))
        processing_times.append(frame_processing_time)
        
        # Keep only last 100 frames for FPS calculation
        if len(processing_times) > 100:
            processing_times.pop(0)
        
        # Display progress
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / max(0.1, elapsed)
            recent_fps = len(processing_times) / max(0.001, sum(processing_times))
            progress = frame_idx / max(1, total_frames) * 100
            
            logger.info(
                f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) | "
                f"FPS: {fps_processing:.1f} (recent: {recent_fps:.1f}) | "
                f"Collisions: {collision_count} | "
                f"Yellow Flags: {potential_count} ({yellow_per_minute:.1f}/min)"
            )
        
        # Write frame to output video if not in heat-map only mode
        if not args.heatmap_only:
            out.write(frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    if not args.heatmap_only:
        out.release()
    
    # Calculate statistics
    total_time = time.time() - start_time
    processing_fps = frame_count / max(0.1, total_time)
    minutes_elapsed = frame_count / (fps * 60)
    yellow_per_minute = potential_count / max(1, minutes_elapsed)
    
    logger.info("\n===== Processing Complete =====")
    logger.info(f"Processed {frame_count} frames in {total_time:.1f} seconds")
    logger.info(f"Processing speed: {processing_fps:.1f} FPS")
    logger.info(f"Collision events detected: {collision_count}")
    logger.info(f"Potential collisions detected: {potential_count}")
    logger.info(f"Yellow flags per minute: {yellow_per_minute:.1f}")
    
    if not args.heatmap_only:
        logger.info(f"Output video saved to: {args.output}")
    
    # CI sanity check
    ci_passed = True
    if yellow_per_minute > CI_MAX_YELLOW_PER_MINUTE:
        logger.warning(f"CI CHECK FAILED: Yellow flags/min ({yellow_per_minute:.1f}) exceeds threshold ({CI_MAX_YELLOW_PER_MINUTE})")
        ci_passed = False
    
    if collision_count < CI_MIN_COLLISIONS and minutes_elapsed >= 0.5:  # Only check if we have at least 30 seconds
        logger.warning(f"CI CHECK FAILED: Only {collision_count} collisions detected (expected {CI_MIN_COLLISIONS}+)")
        ci_passed = False
    
    # Only check FPS if we have sufficient frames
    if frame_count > 200:  
        target_fps = fps * (1 - CI_MAX_FPS_DROP_PCT/100)
        if processing_fps < target_fps:
            logger.warning(f"CI CHECK FAILED: Processing FPS ({processing_fps:.1f}) below target ({target_fps:.1f})")
            ci_passed = False
    
    if ci_passed:
        logger.info("CI SANITY CHECKS: PASSED ✓")
    else:
        logger.warning("CI SANITY CHECKS: FAILED ✗")
        if not args.ignore_ci_checks:
            logger.error("Stopping due to CI check failures. Use --ignore-ci-checks to bypass.")
            return 1  # Return error code
    
    # Generate heat-map analytics
    print("\n===== Generating Heat-map Analytics =====")
    hotspots = heatmap_analyzer.cluster_events()
    
    if hotspots:
        print(f"Identified {len(hotspots)} hotspot clusters")
        heatmap_analyzer.print_analytics_summary()
        
        # Export results
        results = heatmap_analyzer.export_results(background_frame)
        print("\nExported heat-map results:")
        for key, path in results.items():
            print(f"- {key}: {path}")
    else:
        print("No hotspots identified. Try adjusting clustering parameters.")
    
    print("\nDone!")

if __name__ == "__main__":
    # Add ignore-ci-checks argument
    parser.add_argument("--ignore-ci-checks", action="store_true", help="Ignore CI checks that would otherwise stop processing")
    args = parser.parse_args()
    sys.exit(main())
