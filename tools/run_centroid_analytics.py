#!/usr/bin/env python3
"""
Run centroid-based analytics on a verified segment to generate heat-maps.

This script processes a pre-verified segment with centroid-based detection,
focusing on generating analytical heat-maps and hotspot identification.
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
from datetime import datetime
import logging
from ultralytics import YOLO
from collections import deque, defaultdict

# Import collision detection modules
from cooldown_manager import CooldownManager
from heatmap_analytics import HeatmapAnalyzer, NearMissEvent
from collision_config import compute_centroid_threshold
from collision_config import (
    COLLISION_IOU_THRESHOLD, 
    MOTION_CHANGE_THRESHOLD,
    POTENTIAL_COLLISION_THRESHOLD,
    YELLOW_COOLDOWN_TIME
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('centroid_analytics')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Run centroid-based analytics on verified segments")
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--output", type=str, default="centroid_analytics_output", help="Path to output directory")
parser.add_argument("--segment-name", type=str, default=None, help="Segment name for output files")
parser.add_argument("--visualize", action="store_true", help="Generate visualization videos")
parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")
parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0') or 'cpu'")

# Simple object tracker for our analysis
class SimpleTracker:
    def __init__(self, track_id, obj_class, initial_box, max_history=10):
        self.track_id = track_id
        self.obj_class = obj_class
        self.boxes = deque(maxlen=max_history)
        self.boxes.append(initial_box)  # [x1, y1, x2, y2]
        self.centers = deque(maxlen=max_history) 
        center = ((initial_box[0] + initial_box[2]) / 2, (initial_box[1] + initial_box[3]) / 2)
        self.centers.append(center)
        self.last_seen = 0
        
    def update(self, box, frame_idx):
        """Update tracker with new detection."""
        self.boxes.append(box)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.centers.append(center)
        self.last_seen = frame_idx
        
    def get_current_box(self):
        """Get the most recent box."""
        return self.boxes[-1] if self.boxes else None
        
    def get_current_center(self):
        """Get the most recent center."""
        return self.centers[-1] if self.centers else None
        
    def get_velocity_change(self):
        """Simple approximation of velocity change."""
        if len(self.centers) < 3:
            return 0.0
            
        # Calculate velocity from last 3 frames
        prev_vel = np.array(self.centers[-2]) - np.array(self.centers[-3])
        curr_vel = np.array(self.centers[-1]) - np.array(self.centers[-2])
        
        # Magnitude of velocity change
        return np.linalg.norm(curr_vel - prev_vel)

# Simple tracking state manager
class SimpleTrackingManager:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        
    def update(self, detections, frame_idx):
        """Update trackers with new detections using simple nearest-center matching."""
        # If no existing trackers, create new ones for all detections
        if not self.trackers:
            for det in detections:
                box = det['box']
                cls = det['class']
                new_tracker = SimpleTracker(self.next_id, cls, box)
                self.trackers[self.next_id] = new_tracker
                det['track_id'] = self.next_id
                self.next_id += 1
            return detections
            
        # Match detections to existing trackers
        matched_trackers = set()
        matched_detections = []
        
        for det in detections:
            box = det['box']
            cls = det['class']
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Find closest tracker of same class
            best_dist = float('inf')
            best_id = None
            
            for tid, tracker in self.trackers.items():
                if tracker.obj_class != cls:
                    continue
                    
                if tid in matched_trackers:
                    continue
                    
                t_center = tracker.get_current_center()
                if t_center is None:
                    continue
                    
                dist = np.sqrt((center[0] - t_center[0])**2 + (center[1] - t_center[1])**2)
                
                # Simple IOU check
                t_box = tracker.get_current_box()
                iou = self.calculate_iou(box, t_box)
                
                # Use distance for matching but require minimum IoU
                if dist < best_dist and (dist < 50 or iou > 0.2):
                    best_dist = dist
                    best_id = tid
            
            # Update matched tracker or create new one
            if best_id is not None:
                self.trackers[best_id].update(box, frame_idx)
                det['track_id'] = best_id
                matched_trackers.add(best_id)
            else:
                # Create new tracker
                new_tracker = SimpleTracker(self.next_id, cls, box)
                new_tracker.last_seen = frame_idx
                self.trackers[self.next_id] = new_tracker
                det['track_id'] = self.next_id
                self.next_id += 1
                
            matched_detections.append(det)
                
        # Remove trackers not seen in recent frames
        stale_ids = []
        for tid, tracker in self.trackers.items():
            if frame_idx - tracker.last_seen > 30:  # Remove after 1 second at 30 fps
                stale_ids.append(tid)
                
        for tid in stale_ids:
            del self.trackers[tid]
            
        return matched_detections
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        # Determine intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
        
    def get_velocity_change(self, track_id):
        """Get velocity change for a specific tracker."""
        if track_id not in self.trackers:
            return 0.0
        return self.trackers[track_id].get_velocity_change()

# Simple EMA implementation for smoothing
class EMATracker:
    def __init__(self, alpha=0.5):
        self.values = {}
        self.alpha = alpha
        
    def update(self, key, value):
        """Update EMA value for a key."""
        if key not in self.values:
            self.values[key] = {'ema': value, 'history': [value]}
        else:
            ema = self.alpha * value + (1 - self.alpha) * self.values[key]['ema']
            self.values[key]['ema'] = ema
            self.values[key]['history'].append(value)
            
            # Keep history limited
            if len(self.values[key]['history']) > 10:
                self.values[key]['history'] = self.values[key]['history'][-10:]
                
        return self.values[key]['ema']
        
    def get(self, key):
        """Get EMA value for a key."""
        if key not in self.values:
            return 0.0
        return self.values[key]['ema']

def main():
    """Main function for centroid-based analytics."""
    # Parse arguments
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
        
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return 1
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if visualizing
    if args.visualize:
        viz_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.video))[0]}_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(viz_path, fourcc, fps, (width, height))
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
        logger.info(f"Loaded YOLOv8 model for detection")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize our components
    tracking_manager = SimpleTrackingManager()
    ema_tracker = EMATracker(alpha=0.5)  # 0.5 is a good balance between smoothing and responsiveness
    cooldown_manager = CooldownManager(cooldown_time=YELLOW_COOLDOWN_TIME)
    
    # Compute resolution-appropriate centroid threshold
    centroid_threshold = compute_centroid_threshold(width, height)
    logger.info(f"Video resolution: {width}x{height}")
    logger.info(f"Using centroid threshold: {centroid_threshold:.1f}px")
    
    # Initialize heat-map analyzer
    frame_shape = (height, width, 3)
    heatmap_analyzer = HeatmapAnalyzer(
        frame_shape=frame_shape,
        centroid_threshold=centroid_threshold
    )
    
    # Background frame for visualization
    background_frame = None
    
    # Statistics
    start_time = time.time()
    frame_count = 0
    collision_count = 0
    potential_count = 0
    
    # Process each frame
    logger.info(f"Processing video with {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store background frame
        if background_frame is None and frame is not None:
            background_frame = frame.copy()
            
        # Get frame index and timestamp
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_idx / fps
        
        # Run YOLO detection
        results = model(frame, conf=args.confidence, classes=[0, 2, 3, 5, 7])
        
        # Process detections
        detections = []
        for det in results[0].boxes:
            box = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            cls = int(det.cls[0].item())
            conf = float(det.conf[0].item())
            
            # Only keep person and vehicle classes
            if cls == 0 or cls in [2, 3, 5, 7]:
                detections.append({
                    'box': box,
                    'class': cls,
                    'confidence': conf
                })
        
        # Update tracking
        tracked_objects = tracking_manager.update(detections, frame_idx)
        
        # Separate persons and vehicles
        persons = [obj for obj in tracked_objects if obj['class'] == 0]
        vehicles = [obj for obj in tracked_objects if obj['class'] in [2, 3, 5, 7]]
        
        # Create visualization copy if needed
        if args.visualize:
            viz_frame = frame.copy()
            
            # Draw all tracked objects
            for obj in tracked_objects:
                box = obj['box']
                cls = obj['class']
                tid = obj['track_id']
                
                # Different colors for different classes
                if cls == 0:  # Person
                    color = (0, 0, 255)  # Red
                else:  # Vehicle
                    color = (0, 255, 0)  # Green
                    
                # Draw box and ID
                cv2.rectangle(viz_frame, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             color, 2)
                             
                cv2.putText(viz_frame, f"{tid}", 
                           (int(box[0]), int(box[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Track collision flag for this frame
        frame_has_collision = False
        frame_has_potential = False
        
        # Process each person-vehicle pair
        for person in persons:
            p_id = person['track_id']
            p_box = person['box']
            p_center = ((p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2)
            
            for vehicle in vehicles:
                v_id = vehicle['track_id']
                v_box = vehicle['box']
                v_center = ((v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2)
                
                # Calculate IoU
                iou = tracking_manager.calculate_iou(p_box, v_box)
                
                # Calculate centroid distance
                centroid_distance = np.sqrt((p_center[0] - v_center[0])**2 + (p_center[1] - v_center[1])**2)
                
                # Get velocity change
                velocity_change = tracking_manager.get_velocity_change(p_id)
                
                # Update EMA values
                interaction_id = f"p{p_id}_v{v_id}"
                smoothed_iou = ema_tracker.update(f"{interaction_id}_iou", iou)
                smoothed_vel = ema_tracker.update(f"{interaction_id}_vel", velocity_change)
                
                # Collision detection with centroid threshold
                is_collision = (
                    (smoothed_iou > COLLISION_IOU_THRESHOLD and smoothed_vel > MOTION_CHANGE_THRESHOLD) or
                    (centroid_distance < centroid_threshold and iou > 0.02)  # Proximity with minimal overlap
                )
                
                # Potential collision detection with cooldown
                is_potential = False
                if not is_collision:
                    potential_condition = (
                        smoothed_iou > COLLISION_IOU_THRESHOLD * POTENTIAL_COLLISION_THRESHOLD or
                        centroid_distance < centroid_threshold * 1.5  # Wider radius for yellow flags
                    )
                    
                    if potential_condition and not cooldown_manager.in_cooldown(p_id, v_id, current_time):
                        is_potential = True
                        cooldown_manager.set(p_id, v_id, current_time)
                
                # Record event if there's a collision or potential collision
                if is_collision or is_potential:
                    # Create event for heat-map
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
                    
                    # Update counters
                    if is_collision:
                        collision_count += 1
                        frame_has_collision = True
                    else:
                        potential_count += 1
                        frame_has_potential = True
                
                # Draw collision visualization if needed
                if args.visualize and (is_collision or is_potential):
                    color = (0, 0, 255) if is_collision else (0, 255, 255)  # Red for collision, yellow for potential
                    
                    # Draw connecting line
                    cv2.line(viz_frame, 
                           (int(p_center[0]), int(p_center[1])), 
                           (int(v_center[0]), int(v_center[1])),
                           color, 2 if is_collision else 1)
                    
                    # Highlight objects
                    cv2.rectangle(viz_frame, 
                                 (int(p_box[0]), int(p_box[1])), 
                                 (int(p_box[2]), int(p_box[3])), 
                                 color, 3)
                    cv2.rectangle(viz_frame, 
                                 (int(v_box[0]), int(v_box[1])), 
                                 (int(v_box[2]), int(v_box[3])), 
                                 color, 3)
        
        # Add overlays for visualization
        if args.visualize:
            # Add frame info
            cv2.putText(viz_frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
            cv2.putText(viz_frame, f"Collisions: {collision_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            cv2.putText(viz_frame, f"Potential: {potential_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                       
            # Add alert overlays
            if frame_has_collision:
                # Red vignette for collision
                overlay = viz_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 128), -1)
                cv2.addWeighted(overlay, 0.3, viz_frame, 0.7, 0, viz_frame)
                
                # Alert text
                cv2.putText(viz_frame, "COLLISION DETECTED", (width//2 - 150, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                           
            elif frame_has_potential:
                # Yellow vignette for potential
                overlay = viz_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 128, 128), -1)
                cv2.addWeighted(overlay, 0.2, viz_frame, 0.8, 0, viz_frame)
            
            # Write visualization frame
            out.write(viz_frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / max(0.1, elapsed)
            progress = frame_count / max(1, total_frames) * 100
            logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"FPS: {fps_processing:.1f} | "
                      f"Events: {collision_count} collisions, {potential_count} potential")
    
    # Clean up
    cap.release()
    if args.visualize:
        out.release()
    
    # Generate heat-map analytics
    logger.info(f"Processing complete! Generating heat-map analytics...")
    
    # Cluster events to identify hotspots
    hotspots = heatmap_analyzer.cluster_events()
    
    # Print analytics summary
    heatmap_analyzer.print_analytics_summary()
    
    # Export results
    segment_name = args.segment_name or os.path.splitext(os.path.basename(args.video))[0]
    timestamp = int(time.time())
    output_prefix = f"{segment_name}_{timestamp}"
    
    # Set output directory for heat-map analyzer
    heatmap_analyzer.output_dir = args.output
    
    # Export all visualizations
    results = heatmap_analyzer.export_results(background_frame)
    
    # Print stats
    minutes_elapsed = frame_count / (fps * 60)
    total_time = time.time() - start_time
    
    logger.info("\n===== Centroid Analysis Complete =====")
    logger.info(f"Processed {frame_count} frames in {total_time:.1f} seconds")
    logger.info(f"Video duration: {minutes_elapsed:.1f} minutes")
    logger.info(f"Collision events: {collision_count}")
    logger.info(f"Potential collisions: {potential_count}")
    logger.info(f"Events per minute: {(collision_count + potential_count) / max(1, minutes_elapsed):.1f}")
    logger.info(f"Yellow flags per minute: {potential_count / max(1, minutes_elapsed):.1f}")
    
    if hotspots:
        logger.info(f"Identified {len(hotspots)} hotspot clusters")
        logger.info(f"Highest risk score: {hotspots[0].risk_score:.1f}")
        logger.info(f"Results exported to: {args.output}")
    
    return 0
        
if __name__ == "__main__":
    sys.exit(main())
