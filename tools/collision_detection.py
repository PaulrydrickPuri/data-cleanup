#!/usr/bin/env python3
"""
Collision Detection Script for Traffic Safety Analysis

This script analyzes traffic videos to detect and highlight actual collisions 
between pedestrians and vehicles, not just proximity.
"""

import os
import argparse
import logging
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PERSON_CLASS_ID = 0  # Person class ID in COCO
VEHICLE_CLASS_IDS = [0, 2]  # Vehicle and motorcycle in our specialized model
DANGER_THRESHOLD = 150  # Pixel distance for proximity warning
COLLISION_IOU_THRESHOLD = 0.2  # LOWERED from 0.3 to improve recall while maintaining precision
MOTION_CHANGE_THRESHOLD = 20  # LOWERED from 30 to detect more subtle motion changes
# Define threshold for potential collisions (yellow flag state) - 70% of regular threshold
POTENTIAL_COLLISION_THRESHOLD = 0.7
TRACKING_HISTORY_FRAMES = 10  # Number of frames to keep tracking history

# Colors
PERSON_COLOR = (0, 0, 255)  # Red for people
VEHICLE_COLORS = {
    0: (0, 255, 0),  # Green for vehicles
    2: (255, 255, 0)  # Yellow for motorcycles
}
PROXIMITY_COLOR = (0, 165, 255)  # Orange for proximity warning
COLLISION_COLOR = (255, 0, 255)  # Purple for collision detection
POTENTIAL_COLOR = (0, 255, 255)  # Yellow for potential collision detection

class ObjectTracker:
    """Tracker for objects across frames to analyze motion patterns."""
    
    def __init__(self, object_id, object_type, initial_box, max_history=TRACKING_HISTORY_FRAMES):
        self.object_id = object_id
        self.object_type = object_type  # 'person' or 'vehicle'
        self.boxes = deque(maxlen=max_history)
        self.centers = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history-1)
        self.add_detection(initial_box)
        self.collision_state = None  # None, 'proximity', 'collision', 'post_collision'
        self.collision_frame = -1  # Frame number when collision occurred
        self.last_seen = 0  # Last frame this object was detected
    
    def add_detection(self, box):
        """Add new detection and update tracking history."""
        self.boxes.append(box)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.centers.append(center)
        
        # Calculate velocity if we have multiple centers
        if len(self.centers) >= 2:
            prev_center = self.centers[-2]
            velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
            self.velocities.append(velocity)
    
    def get_current_box(self):
        """Get the most recent bounding box."""
        if self.boxes:
            return self.boxes[-1]
        return None
    
    def get_current_center(self):
        """Get the most recent center position."""
        if self.centers:
            return self.centers[-1]
        return None
    
    def get_average_velocity(self, frames=3):
        """Calculate average velocity over recent frames."""
        if len(self.velocities) == 0:
            return (0, 0)
        
        frames = min(frames, len(self.velocities))
        recent_velocities = list(self.velocities)[-frames:]
        avg_vx = sum(v[0] for v in recent_velocities) / frames
        avg_vy = sum(v[1] for v in recent_velocities) / frames
        return (avg_vx, avg_vy)
    
    def get_velocity_change(self):
        """Calculate sudden change in velocity (acceleration)."""
        if len(self.velocities) < 2:
            return 0
        
        v_current = self.velocities[-1]
        v_prev = self.velocities[-2]
        
        # Calculate magnitude of velocity change
        delta_vx = v_current[0] - v_prev[0]
        delta_vy = v_current[1] - v_prev[1]
        change = np.sqrt(delta_vx**2 + delta_vy**2)
        
        # Guard against NaN (improved numerical stability)
        if np.isnan(change):
            return 0
            
        return change
    
    def is_motion_consistent(self):
        """Check if the motion is consistent or has sudden changes."""
        if len(self.velocities) < 3:
            return True
        
        velocity_change = self.get_velocity_change()
        return velocity_change < MOTION_CHANGE_THRESHOLD

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    # Box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    return iou

def generate_unique_id():
    """Generate a unique ID for tracking objects."""
    generate_unique_id.counter += 1
    return generate_unique_id.counter
generate_unique_id.counter = 0

def detect_collisions(video_path, general_model_path, vehicle_model_path, output_path=None, 
                      conf_threshold=0.25, iou_threshold=0.45, device='auto'):
    """
    Analyze a traffic video to detect and visualize collisions between pedestrians and vehicles.
    
    Args:
        video_path: Path to the input video file
        general_model_path: Path to the COCO YOLOv8 model for people detection
        vehicle_model_path: Path to the specialized vehicle detection model
        output_path: Path to save the output video (default: auto-generated)
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        device: Device to use for inference ('cpu', 'cuda', or 'auto')
    """
    # Check device availability
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Starting collision detection analysis using device: {device}")
    
    # Load models
    logger.info(f"Loading general COCO model: {general_model_path}")
    general_model = YOLO(general_model_path)
    
    logger.info(f"Loading specialized vehicle model: {vehicle_model_path}")
    vehicle_model = YOLO(vehicle_model_path)
    
    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Generate output path if not provided
    if output_path is None:
        video_name = Path(video_path).stem
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"{video_name}_collision_analysis_{timestamp}.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Object trackers
    person_trackers = {}  # id -> ObjectTracker
    vehicle_trackers = {}  # id -> ObjectTracker
    
    # Collision statistics
    collision_stats = {
        'total_frames': 0,
        'frames_with_people': 0,
        'frames_with_vehicles': 0,
        'proximity_warnings': 0,
        'collision_events': 0,
        'people_involved': set(),
        'vehicles_involved': set()
    }
    
    # Collision event log
    collision_events = []
    
    # Process video
    frame_count = 0
    start_time = time.time()
    
    try:
        with torch.no_grad():  # Disable gradient calculation for faster inference
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                collision_stats['total_frames'] += 1
                
                # Log progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    remaining = (total_frames - frame_count) * elapsed / frame_count
                    logger.info(f"Processing frame {frame_count}/{total_frames} "
                               f"({frame_count/total_frames*100:.1f}%) - "
                               f"Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
                
                # Run people detection with general COCO model
                general_results = general_model(frame, conf=conf_threshold, iou=iou_threshold, 
                                              classes=[PERSON_CLASS_ID], device=device)[0]
                
                # Run vehicle detection with specialized model
                vehicle_results = vehicle_model(frame, conf=conf_threshold, iou=iou_threshold, 
                                              device=device)[0]
                
                # Create a copy of the frame for visualization
                result_frame = frame.copy()
                
                # Process person detections
                current_people = []
                has_people = False
                
                for result in general_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = result
                    if int(class_id) == PERSON_CLASS_ID and conf >= conf_threshold:
                        has_people = True
                        box = (int(x1), int(y1), int(x2), int(y2))
                        current_people.append(box)
                
                if has_people:
                    collision_stats['frames_with_people'] += 1
                
                # Process vehicle detections
                current_vehicles = []
                has_vehicles = False
                
                for result in vehicle_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = result
                    if int(class_id) in VEHICLE_CLASS_IDS and conf >= conf_threshold:
                        has_vehicles = True
                        box = (int(x1), int(y1), int(x2), int(y2))
                        vehicle_type = int(class_id)
                        current_vehicles.append((box, vehicle_type))
                
                if has_vehicles:
                    collision_stats['frames_with_vehicles'] += 1
                
                # Updated tracking logic (simplified association based on IoU)
                # Update person trackers
                matched_person_ids = set()
                
                for box in current_people:
                    best_iou = 0.5  # Minimum IoU to consider a match
                    best_id = None
                    
                    for person_id, tracker in person_trackers.items():
                        if person_id in matched_person_ids:
                            continue
                        
                        prev_box = tracker.get_current_box()
                        iou = calculate_iou(box, prev_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_id = person_id
                    
                    if best_id is not None:
                        # Update existing tracker
                        person_trackers[best_id].add_detection(box)
                        person_trackers[best_id].last_seen = frame_count
                        matched_person_ids.add(best_id)
                    else:
                        # Create new tracker
                        new_id = generate_unique_id()
                        person_trackers[new_id] = ObjectTracker(new_id, 'person', box)
                        person_trackers[new_id].last_seen = frame_count
                        matched_person_ids.add(new_id)
                
                # Update vehicle trackers
                matched_vehicle_ids = set()
                
                for box, vehicle_type in current_vehicles:
                    best_iou = 0.5  # Minimum IoU to consider a match
                    best_id = None
                    
                    for vehicle_id, tracker in vehicle_trackers.items():
                        if vehicle_id in matched_vehicle_ids:
                            continue
                        
                        prev_box = tracker.get_current_box()
                        iou = calculate_iou(box, prev_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_id = vehicle_id
                    
                    if best_id is not None:
                        # Update existing tracker
                        vehicle_trackers[best_id].add_detection(box)
                        vehicle_trackers[best_id].last_seen = frame_count
                        matched_vehicle_ids.add(best_id)
                    else:
                        # Create new tracker
                        new_id = generate_unique_id()
                        vehicle_trackers[new_id] = ObjectTracker(new_id, 'vehicle', box)
                        vehicle_trackers[new_id].last_seen = frame_count
                        matched_vehicle_ids.add(new_id)
                
                # Remove stale trackers (not seen for 10 frames)
                stale_threshold = 10
                for tracker_dict in [person_trackers, vehicle_trackers]:
                    ids_to_remove = []
                    for obj_id, tracker in tracker_dict.items():
                        if frame_count - tracker.last_seen > stale_threshold:
                            ids_to_remove.append(obj_id)
                    
                    for obj_id in ids_to_remove:
                        del tracker_dict[obj_id]
                
                # Detect collisions and proximity warnings
                frame_has_proximity = False
                frame_has_collision = False
                frame_has_potential_collision = False
                
                for person_id, person_tracker in person_trackers.items():
                    if frame_count - person_tracker.last_seen > 0:
                        continue  # Skip if person not seen in current frame
                    
                    person_box = person_tracker.get_current_box()
                    person_center = person_tracker.get_current_center()
                    
                    for vehicle_id, vehicle_tracker in vehicle_trackers.items():
                        if frame_count - vehicle_tracker.last_seen > 0:
                            continue  # Skip if vehicle not seen in current frame
                        
                        vehicle_box = vehicle_tracker.get_current_box()
                        vehicle_center = vehicle_tracker.get_current_center()
                        
                        # Calculate distance between person and vehicle centers
                        distance = np.sqrt((person_center[0] - vehicle_center[0])**2 + 
                                         (person_center[1] - vehicle_center[1])**2)
                        
                        # Calculate IoU between person and vehicle
                        box_iou = calculate_iou(person_box, vehicle_box)
                        
                        # Get velocity changes
                        person_vel_change = person_tracker.get_velocity_change()
                        vehicle_vel_change = vehicle_tracker.get_velocity_change()
                        
                        # Normalize velocity change for scoring (0-1 range)
                        velocity_change_norm = min(1.0, person_vel_change / 100.0)
                        
                        # Calculate combined collision score - giving IoU more weight (70%) than before
                        collision_score = box_iou * 0.7 + velocity_change_norm * 0.3
                        
                        # Collision detection criteria:
                        # 1. High IoU between boxes
                        # 2. Sudden change in person velocity
                        # 3. Sufficient tracking history
                        is_collision = (
                            box_iou > COLLISION_IOU_THRESHOLD and 
                            person_vel_change > MOTION_CHANGE_THRESHOLD and
                            len(person_tracker.velocities) >= 3
                        )
                        
                        # Potential collision detection - lower threshold (yellow flag)
                        # Either a meaningful IoU overlap OR a significant motion change
                        is_potential_collision = (
                            not is_collision and (
                                box_iou > COLLISION_IOU_THRESHOLD * POTENTIAL_COLLISION_THRESHOLD or
                                person_vel_change > MOTION_CHANGE_THRESHOLD * POTENTIAL_COLLISION_THRESHOLD
                            ) and len(person_tracker.velocities) >= 3
                        )
                        
                        # If already in collision state, maintain it for a few frames
                        if person_tracker.collision_state == 'collision':
                            frames_since_collision = frame_count - person_tracker.collision_frame
                            if frames_since_collision < 15:  # Keep the collision state for 15 frames
                                is_collision = True
                        
                        # If already in potential collision state, maintain it for a few frames
                        if person_tracker.collision_state == 'potential':
                            frames_since_potential_collision = frame_count - person_tracker.collision_frame
                            if frames_since_potential_collision < 30:  # Keep the potential collision state for 30 frames
                                is_potential_collision = True
                            else:
                                person_tracker.collision_state = 'normal'
                        
                        # Proximity warning
                        is_proximity = distance < DANGER_THRESHOLD
                        
                        # Update tracking state
                        if is_collision:
                            if person_tracker.collision_state != 'collision':
                                # New collision detected
                                person_tracker.collision_state = 'collision'
                                person_tracker.collision_frame = frame_count
                                collision_stats['collision_events'] += 1
                                collision_stats['people_involved'].add(person_id)
                                collision_stats['vehicles_involved'].add(vehicle_id)
                                
                                # Log the collision event
                                collision_events.append({
                                    'frame': frame_count,
                                    'person_id': person_id,
                                    'vehicle_id': vehicle_id,
                                    'box_iou': box_iou,
                                    'velocity_change': person_vel_change,
                                    'score': collision_score
                                })
                            
                            frame_has_collision = True
                            
                            # Draw collision visualization
                            p_box = person_tracker.get_current_box()
                            v_box = vehicle_tracker.get_current_box()
                            
                            # Highlight the boxes with thicker outline
                            cv2.rectangle(result_frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), 
                                        COLLISION_COLOR, 3)
                            cv2.rectangle(result_frame, (v_box[0], v_box[1]), (v_box[2], v_box[3]), 
                                        COLLISION_COLOR, 3)
                            
                            # Draw connection line
                            cv2.line(result_frame, 
                                   (int(person_center[0]), int(person_center[1])), 
                                   (int(vehicle_center[0]), int(vehicle_center[1])), 
                                   COLLISION_COLOR, 2)
                            
                            # Add collision text with score
                            cv2.putText(result_frame, 
                                      f"COLLISION DETECTED! Score: {collision_score:.2f}", 
                                      (int((person_center[0] + vehicle_center[0])/2) - 120, 
                                       int((person_center[1] + vehicle_center[1])/2) - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLLISION_COLOR, 2)
                        
                        # NEW: Check for potential collision (yellow flag state)
                        elif is_potential_collision:
                            if person_tracker.collision_state != 'potential':
                                person_tracker.collision_state = 'potential'
                                collision_stats.setdefault('potential_collisions', 0)
                                collision_stats['potential_collisions'] += 1
                            
                            frame_has_potential_collision = True
                            
                            # Draw potential collision visualization
                            p_box = person_tracker.get_current_box()
                            v_box = vehicle_tracker.get_current_box()
                            
                            # Highlight boxes with medium outline
                            cv2.rectangle(result_frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), 
                                        POTENTIAL_COLOR, 2)
                            cv2.rectangle(result_frame, (v_box[0], v_box[1]), (v_box[2], v_box[3]), 
                                        POTENTIAL_COLOR, 2)
                            
                            # Draw connection line
                            cv2.line(result_frame, 
                                   (int(person_center[0]), int(person_center[1])), 
                                   (int(vehicle_center[0]), int(vehicle_center[1])), 
                                   POTENTIAL_COLOR, 1)
                            
                            # Add potential collision text with score
                            cv2.putText(result_frame, 
                                      f"POTENTIAL COLLISION: {collision_score:.2f}", 
                                      (int((person_center[0] + vehicle_center[0])/2) - 120, 
                                       int((person_center[1] + vehicle_center[1])/2) - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, POTENTIAL_COLOR, 2)
                                      
                        elif is_proximity:
                            frame_has_proximity = True
                            person_tracker.collision_state = 'proximity'
                            collision_stats['proximity_warnings'] += 1
                            
                            # Only draw proximity warning if there's no collision or potential collision
                            if not frame_has_collision and not frame_has_potential_collision:
                                # Draw proximity warning
                                cv2.line(result_frame, 
                                       (int(person_center[0]), int(person_center[1])), 
                                       (int(vehicle_center[0]), int(vehicle_center[1])), 
                                       PROXIMITY_COLOR, 2)
                                
                                # Add proximity text
                                cv2.putText(result_frame, 
                                          f"PROXIMITY WARNING: {distance:.1f}px", 
                                          (int((person_center[0] + vehicle_center[0])/2) - 80, 
                                           int((person_center[1] + vehicle_center[1])/2) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, PROXIMITY_COLOR, 2)
                
                # Draw all tracked objects
                for person_id, tracker in person_trackers.items():
                    if frame_count - tracker.last_seen > 0:
                        continue
                    
                    box = tracker.get_current_box()
                    if tracker.collision_state != 'collision':  # Skip if already drawn for collision
                        color = PERSON_COLOR
                        cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        
                        # Add person ID and tracking status
                        label = f"Person #{person_id}"
                        cv2.putText(result_frame, label, (box[0], box[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw motion trail
                    if len(tracker.centers) > 1:
                        centers = list(tracker.centers)
                        for i in range(1, len(centers)):
                            cv2.line(result_frame, 
                                   (int(centers[i-1][0]), int(centers[i-1][1])), 
                                   (int(centers[i][0]), int(centers[i][1])), 
                                   PERSON_COLOR, 1)
                
                for vehicle_id, tracker in vehicle_trackers.items():
                    if frame_count - tracker.last_seen > 0:
                        continue
                    
                    box = tracker.get_current_box()
                    if tracker.collision_state != 'collision':  # Skip if already drawn for collision
                        color = VEHICLE_COLORS.get(0, (255, 0, 255))  # Default to magenta
                        cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        
                        # Add vehicle ID
                        label = f"Vehicle #{vehicle_id}"
                        cv2.putText(result_frame, label, (box[0], box[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add frame info overlay
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add detection counts
                cv2.putText(result_frame, 
                          f"People: {len([p for p, t in person_trackers.items() if frame_count - t.last_seen <= 0])}  " + 
                          f"Vehicles: {len([v for v, t in vehicle_trackers.items() if frame_count - t.last_seen <= 0])}", 
                          (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add collision alert
                if frame_has_collision:
                    cv2.putText(result_frame, "ALERT: COLLISION DETECTED!", 
                              (width//2 - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLLISION_COLOR, 2)
                elif frame_has_potential_collision:
                    cv2.putText(result_frame, "WARNING: POTENTIAL COLLISION!", 
                              (width//2 - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, POTENTIAL_COLOR, 2)
                elif frame_has_proximity:
                    cv2.putText(result_frame, "CAUTION: Objects in Proximity", 
                              (width//2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PROXIMITY_COLOR, 2)
                
                # Write frame to output video
                out.write(result_frame)
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        
        # Calculate and print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info("=== Collision Analysis Complete ===")
        logger.info(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
        logger.info(f"Frames with people: {collision_stats['frames_with_people']} ({collision_stats['frames_with_people']/collision_stats['total_frames']*100:.1f}%)")
        logger.info(f"Frames with vehicles: {collision_stats['frames_with_vehicles']} ({collision_stats['frames_with_vehicles']/collision_stats['total_frames']*100:.1f}%)")
        logger.info(f"Proximity warnings: {collision_stats['proximity_warnings']}")
        logger.info(f"Collision events detected: {collision_stats['collision_events']}")
        potential_collisions = collision_stats.get('potential_collisions', 0)
        logger.info(f"Potential collisions detected: {potential_collisions}")
        logger.info(f"People involved in collisions: {len(collision_stats['people_involved'])}")
        logger.info(f"Vehicles involved in collisions: {len(collision_stats['vehicles_involved'])}")
        logger.info(f"Output saved to: {output_path}")
        
        # Print collision event details
        if collision_events:
            logger.info("\nCollision Event Details:")
            for i, event in enumerate(collision_events):
                logger.info(f"Event #{i+1} at frame {event['frame']}: " +
                           f"Person #{event['person_id']} and Vehicle #{event['vehicle_id']} - " +
                           f"Score: {event.get('score', 0):.3f}, IoU: {event['box_iou']:.3f}, " +
                           f"Velocity change: {event['velocity_change']:.2f}")
        
        return output_path, collision_stats, collision_events

def main():
    """Parse command line arguments and run the collision detection analysis."""
    parser = argparse.ArgumentParser(description="Collision Detection for Traffic Safety Analysis")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--general-model", default="yolov8n.pt", 
                      help="Path to the COCO YOLOv8 model for people detection")
    parser.add_argument("--vehicle-model", default="models/outputs/vehicle_detection2/weights/best.pt", 
                      help="Path to the specialized vehicle detection model")
    parser.add_argument("--output", help="Path to save the output video")
    parser.add_argument("--conf", type=float, default=0.25, 
                      help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45, 
                      help="IoU threshold for NMS")
    parser.add_argument("--device", default="auto", 
                      help="Device to use for inference ('cpu', 'cuda', or 'auto')")
    
    args = parser.parse_args()
    
    # Print arguments
    logger.info("=== Collision Detection Configuration ===")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"{arg_name}: {arg_value}")
    
    # Run collision detection
    detect_collisions(
        video_path=args.video,
        general_model_path=args.general_model,
        vehicle_model_path=args.vehicle_model,
        output_path=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

if __name__ == "__main__":
    main()
