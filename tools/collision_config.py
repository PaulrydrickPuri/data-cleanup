#!/usr/bin/env python3
"""
Configuration constants and functions for collision detection
"""
import numpy as np

# Base detection thresholds - tuned for persistent ID tracking
COLLISION_IOU_THRESHOLD = 0.10      # Lowered from 0.12 - recall-first
MOTION_CHANGE_THRESHOLD = 10        # Lowered from 12 - recall-first

# Default centroid threshold - will be automatically scaled with resolution
CENTROID_THRESHOLD = 80             # For 720p video

# Potential collision parameters
POTENTIAL_COLLISION_THRESHOLD = 0.60  # Lowered from 0.70 - wider net for near-misses
YELLOW_COOLDOWN_TIME = 2.0           # Cooldown to reduce yellow flag spam


def compute_centroid_threshold(width, height):
    """
    Automatically compute appropriate centroid threshold based on video dimensions.
    Scales threshold proportionally to the diagonal of the frame.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
    
    Returns:
        Resolution-appropriate centroid threshold
    """
    # Base calculation on diagonal of the frame (Euclidean norm of dimensions)
    diagonal = (width**2 + height**2)**0.5
    
    # 720p reference: sqrt(1280^2 + 720^2) ≈ 1468px diagonal
    # At 720p, we want threshold ≈ 80px
    reference_diagonal = 1468.0  # 720p diagonal
    reference_threshold = 80.0   # Threshold for 720p
    
    # Scale threshold linearly with diagonal
    # Use 0.055 * diagonal as a general rule (roughly equivalent to 80px at 720p)
    scaled_threshold = 0.055 * diagonal
    
    # Ensure threshold is at least 40px for very small videos
    # and not excessively large for 4K+ videos
    return max(40, min(200, scaled_threshold))

# EMA smoothing parameters
EMA_ALPHA = 0.5  # Lowered from 0.7 - snappier, half-life ≈ 1 frame
EMA_WINDOW = 5   # Number of frames for EMA calculation

# Scene-adaptive threshold parameters
SCENE_HISTORY_FRAMES = 30  # Frames to consider for scene dynamics
SCENE_ALPHA = 0.8          # Scene EMA smoothing factor

# Visualization colors
PERSON_COLOR = (0, 0, 255)       # Red for people
VEHICLE_COLORS = {
    0: (0, 255, 0),              # Green for vehicles
    2: (255, 255, 0)             # Yellow for motorcycles
}
PROXIMITY_COLOR = (0, 165, 255)  # Orange for proximity warning
COLLISION_COLOR = (255, 0, 255)  # Purple for collision detection
POTENTIAL_COLOR = (0, 255, 255)  # Yellow for potential collision

# Heat map parameters
HEATMAP_RADIUS = 50              # Pixel radius for heat map points
DBSCAN_EPSILON = 30              # DBSCAN clustering distance threshold
DBSCAN_MIN_SAMPLES = 3           # Minimum samples for DBSCAN clustering

# Helper functions for adaptive thresholds
def get_adaptive_thresholds(scene_speed, prev_iou_threshold=None, prev_motion_threshold=None):
    """
    Calculate adaptive thresholds based on scene motion.
    
    Args:
        scene_speed: Median speed in the scene
        prev_iou_threshold: Previous IoU threshold
        prev_motion_threshold: Previous motion threshold
    
    Returns:
        Tuple of (iou_threshold, motion_threshold)
    """
    # Calculate dynamic thresholds based on scene speed
    # Higher scene speed = higher thresholds
    dyn_iou = 0.15 + 0.25 * np.clip(scene_speed / 50.0, 0, 1)  # Range: 0.15 -> 0.40
    dyn_motion = 15 + 20 * np.clip(scene_speed / 50.0, 0, 1)   # Range: 15 -> 35
    
    # Apply EMA smoothing if previous values provided
    if prev_iou_threshold is not None and prev_motion_threshold is not None:
        dyn_iou = SCENE_ALPHA * prev_iou_threshold + (1-SCENE_ALPHA) * dyn_iou
        dyn_motion = SCENE_ALPHA * prev_motion_threshold + (1-SCENE_ALPHA) * dyn_motion
    
    return dyn_iou, dyn_motion

# Dictionary to store EMA values for each object pair
_ema_storage = {}

# Telemetry storage for debugging
telemetry_dump = []

def update_ema_values(obj_id, iou_value, vel_value):
    """
    Update EMA values for an object.
    
    Args:
        obj_id: Object ID
        iou_value: Current IoU value
        vel_value: Current velocity change value
    
    Returns:
        Tuple of (smoothed_iou, smoothed_vel)
    """
    # Initialize if not present
    if obj_id not in _ema_storage:
        _ema_storage[obj_id] = {
            'iou_history': [iou_value],
            'vel_history': [vel_value],
            'iou_ema': iou_value,
            'vel_ema': vel_value
        }
        return iou_value, vel_value
    
    # Get object storage
    obj_store = _ema_storage[obj_id]
    
    # Add new values to history
    obj_store['iou_history'].append(iou_value)
    obj_store['vel_history'].append(vel_value)
    
    # Limit history length
    if len(obj_store['iou_history']) > EMA_WINDOW:
        obj_store['iou_history'].pop(0)
    if len(obj_store['vel_history']) > EMA_WINDOW:
        obj_store['vel_history'].pop(0)
    
    # Update EMA values
    obj_store['iou_ema'] = EMA_ALPHA * obj_store['iou_ema'] + (1-EMA_ALPHA) * iou_value
    obj_store['vel_ema'] = EMA_ALPHA * obj_store['vel_ema'] + (1-EMA_ALPHA) * vel_value
    
    return obj_store['iou_ema'], obj_store['vel_ema']

def get_collision_state(iou, velocity_change, smoothed_iou=None, smoothed_vel=None,
                       p_center=None, v_center=None, cooldown_manager=None, current_time=None,
                       telemetry_storage=None, frame_number=None, person_id=None, vehicle_id=None):
    """
    Determine if there is a collision or potential collision based on IoU, velocity change,
    and centroid distance
    
    Args:
        iou: Intersection over Union value
        velocity_change: Magnitude of velocity change
        smoothed_iou: Smoothed IoU value (optional)
        smoothed_vel: Smoothed velocity change value (optional)
        p_center: Person center point (x, y) (required for centroid detection)
        v_center: Vehicle center point (x, y) (required for centroid detection)
        cooldown_manager: Manager for yellow flag cooldowns (optional)
        current_time: Current time for cooldown management (optional)
        telemetry_storage: Storage for telemetry data (optional)
        frame_number: Current frame number for telemetry (optional)
        person_id: ID of the person for telemetry (optional)
        vehicle_id: ID of the vehicle for telemetry (optional)
    
    Returns:
        Tuple of (is_collision, is_potential_collision)
    """
    # Calculate centroid distance if centers are provided
    centroid_distance = None
    if p_center is not None and v_center is not None:
        centroid_distance = np.hypot(p_center[0] - v_center[0], p_center[1] - v_center[1])
    
    # If smoothed values are provided, use enhanced detection
    if smoothed_iou is not None and smoothed_vel is not None and centroid_distance is not None:
        iou_threshold = COLLISION_IOU_THRESHOLD
        motion_threshold = MOTION_CHANGE_THRESHOLD
        
        # Collision if (IoU∧Vel) OR (close centroids with tiny overlap)
        is_collision = (
            (smoothed_iou > iou_threshold and smoothed_vel > motion_threshold) or
            (centroid_distance < CENTROID_THRESHOLD and iou > 0.02)
        )
        
        # Potential collision with cooldown
        is_potential = False
        if not is_collision:
            potential_condition = (
                smoothed_iou > iou_threshold * POTENTIAL_COLLISION_THRESHOLD or
                centroid_distance < CENTROID_THRESHOLD * 1.5
            )
            
            # Apply cooldown to yellow flags if manager is provided
            if potential_condition and cooldown_manager is not None and current_time is not None:
                if not cooldown_manager.in_cooldown(person_id, vehicle_id, current_time):
                    is_potential = True
                    cooldown_manager.set(person_id, vehicle_id, current_time)
            else:
                # Legacy behavior if no cooldown manager
                is_potential = potential_condition
        
        # Store telemetry data if provided
        if telemetry_storage is not None and frame_number is not None:
            telemetry_storage.add_entry(
                frame_number, 
                person_id,
                vehicle_id,
                iou, 
                velocity_change, 
                smoothed_iou, 
                smoothed_vel,
                iou_threshold,
                motion_threshold,
                centroid_distance < CENTROID_THRESHOLD,
                is_collision
            )
    else:
        # Legacy detection using raw values
        is_collision = (
            iou > COLLISION_IOU_THRESHOLD and 
            velocity_change > MOTION_CHANGE_THRESHOLD
        )
        is_potential = not is_collision and (
            iou > COLLISION_IOU_THRESHOLD * POTENTIAL_COLLISION_THRESHOLD
        )
    
    return is_collision, is_potential

def reset_ema_storage():
    """Reset EMA storage dictionary and telemetry dump."""
    global _ema_storage, telemetry_dump
    _ema_storage = {}
    telemetry_dump = []
