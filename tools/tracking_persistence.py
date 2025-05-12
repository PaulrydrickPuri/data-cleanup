#!/usr/bin/env python3
"""
Hungarian algorithm-based tracking persistence for collision detection
Maintains object identity across frames to reduce false positives from tracking
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes.
    Boxes in format [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    # Compute the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def calculate_distance(center1, center2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def transfer_object_history(source_tracker, target_tracker):
    """Transfer relevant history from source tracker to target tracker."""
    # Keep the target's current box and center
    current_box = target_tracker.get_current_box()
    current_center = target_tracker.get_current_center()
    
    # Transfer the object ID and collision state
    target_tracker.object_id = source_tracker.object_id
    target_tracker.collision_state = source_tracker.collision_state
    target_tracker.collision_frame = source_tracker.collision_frame
    
    # Only update velocity history if it makes sense
    if len(source_tracker.velocities) > 0:
        # Add the previous boxes and centers
        max_to_transfer = min(len(source_tracker.boxes), target_tracker.boxes.maxlen - 1)
        for i in range(max_to_transfer):
            target_tracker.boxes.appendleft(source_tracker.boxes[i])
            target_tracker.centers.appendleft(source_tracker.centers[i])
        
        # Add the previous velocities
        max_vel_to_transfer = min(len(source_tracker.velocities), target_tracker.velocities.maxlen - 1)
        for i in range(max_vel_to_transfer):
            target_tracker.velocities.appendleft(source_tracker.velocities[i])
    
    # Add the current detection at the end
    target_tracker.boxes.append(current_box)
    target_tracker.centers.append(current_center)
    
    # Calculate and add the newest velocity if we have enough history
    if len(target_tracker.centers) >= 2:
        current = target_tracker.centers[-1]
        previous = target_tracker.centers[-2]
        velocity = (current[0] - previous[0], current[1] - previous[1])
        target_tracker.velocities.append(velocity)
    
    return target_tracker

def maintain_tracking_ids(previous_trackers, current_detections, tracker_class, obj_type, frame_idx, 
                         max_distance=50, min_iou=0.2, debug=False):
    """
    Maintain object identity across frames using Hungarian algorithm.
    
    Args:
        previous_trackers: Dictionary of trackers from previous frame {id: tracker}
        current_detections: List of bounding boxes in current frame [x1, y1, x2, y2, conf, class]
        tracker_class: Class to instantiate for new trackers
        obj_type: Type of object ('person' or 'vehicle')
        frame_idx: Current frame index
        max_distance: Maximum distance for association
        min_iou: Minimum IoU for association
        debug: Whether to print debug info
        
    Returns:
        Dictionary of updated trackers for current frame
    """
    # Skip if no previous trackers or no current detections
    if not previous_trackers:
        # Create new trackers for all detections
        new_trackers = {}
        for i, detection in enumerate(current_detections):
            box = detection[:4]  # [x1, y1, x2, y2]
            obj_id = i  # Simple ID for new objects
            new_trackers[obj_id] = tracker_class(obj_id, obj_type, box)
            new_trackers[obj_id].last_seen = frame_idx
        return new_trackers
        
    if not current_detections:
        return {}
    
    # Extract previous tracker data
    prev_ids = list(previous_trackers.keys())
    prev_trackers = list(previous_trackers.values())
    
    # Prepare current detection boxes
    curr_boxes = [det[:4] for det in current_detections]  # [x1, y1, x2, y2]
    
    # Calculate cost matrix based on IoU and distance
    cost_matrix = np.zeros((len(prev_trackers), len(curr_boxes)))
    iou_matrix = np.zeros((len(prev_trackers), len(curr_boxes)))
    
    for i, tracker in enumerate(prev_trackers):
        prev_box = tracker.get_current_box()
        prev_center = tracker.get_current_center()
        
        for j, curr_box in enumerate(curr_boxes):
            curr_center = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
            
            # Calculate IoU
            iou = calculate_iou(prev_box, curr_box)
            iou_matrix[i, j] = iou
            
            # Calculate center distance
            distance = calculate_distance(prev_center, curr_center)
            
            # Combined cost: 1 - IoU + normalized distance
            # Higher IoU means lower cost, higher distance means higher cost
            cost_matrix[i, j] = (1 - iou) + (distance / max_distance)
    
    # Apply Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create new trackers dictionary
    new_trackers = {}
    matched_detections = set()
    
    # Process matched pairs
    for i, j in zip(row_ind, col_ind):
        # Check if match is valid (below distance threshold and above IoU threshold)
        prev_tracker = prev_trackers[i]
        curr_box = curr_boxes[j]
        
        # Only accept matches with reasonable IoU or distance
        if (iou_matrix[i, j] > min_iou or 
            cost_matrix[i, j] < 1.5):  # Cost threshold from tuning
            
            # Create new tracker with current detection
            new_tracker = tracker_class(prev_ids[i], obj_type, curr_box)
            new_tracker.last_seen = frame_idx
            
            # Transfer history from previous tracker
            new_tracker = transfer_object_history(prev_tracker, new_tracker)
            
            # Add to new trackers
            new_trackers[prev_ids[i]] = new_tracker
            matched_detections.add(j)
            
            if debug and obj_type == 'person':
                logger.debug(f"Maintained ID {prev_ids[i]} with IoU {iou_matrix[i, j]:.2f}")
    
    # Process unmatched detections
    next_id = max(prev_ids) + 1 if prev_ids else 0
    for j in range(len(curr_boxes)):
        if j not in matched_detections:
            # Create new tracker with fresh ID
            curr_box = curr_boxes[j]
            new_tracker = tracker_class(next_id, obj_type, curr_box)
            new_tracker.last_seen = frame_idx
            
            # Add to new trackers
            new_trackers[next_id] = new_tracker
            next_id += 1
            
            if debug and obj_type == 'person':
                logger.debug(f"Created new ID {next_id-1}")
    
    if debug:
        logger.debug(f"Frame {frame_idx}: {len(prev_trackers)} previous {obj_type}s, " + 
                    f"{len(curr_boxes)} current detections, {len(new_trackers)} after matching")
    
    return new_trackers

# Helper for dynamic thresholds
def get_adaptive_thresholds(scene_motion, prev_iou_threshold, prev_motion_threshold, 
                           alpha=0.8, base_iou=0.2, base_motion=20):
    """
    Calculate adaptive thresholds based on scene motion.
    
    Args:
        scene_motion: Average motion in the scene
        prev_iou_threshold: Previous IoU threshold
        prev_motion_threshold: Previous motion threshold
        alpha: EMA smoothing factor
        base_iou: Base IoU threshold
        base_motion: Base motion threshold
    
    Returns:
        Tuple of (iou_threshold, motion_threshold)
    """
    # Calculate dynamic thresholds with EMA smoothing
    dynamic_iou = alpha * prev_iou_threshold + (1-alpha) * (base_iou + scene_motion / 500.0)
    dynamic_iou = np.clip(dynamic_iou, 0.15, 0.35)  # Reasonable bounds
    
    dynamic_motion = alpha * prev_motion_threshold + (1-alpha) * (base_motion + scene_motion * 0.2)
    dynamic_motion = np.clip(dynamic_motion, 10, 40)  # Reasonable bounds
    
    return dynamic_iou, dynamic_motion
