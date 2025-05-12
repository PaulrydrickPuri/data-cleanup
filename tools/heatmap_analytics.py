#!/usr/bin/env python3
"""
Heat-map analytics for collision and near-miss visualization.
Uses DBSCAN clustering to identify hotspots of potential collision events.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Circle
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Heat-map configuration
HEATMAP_CONFIG = {
    'eps': None,              # DBSCAN epsilon (dynamically set based on CENTROID_THRESHOLD)
    'min_samples': 3,        # Min points to form a cluster
    'alpha': 0.7,            # Transparency for heat overlay
    'resolution': 1.0,       # Resolution multiplier for heat-map (higher = more detailed)
    'min_events': 5,         # Minimum events to consider a valid hotspot
    'gaussian_sigma': 25,    # Sigma for Gaussian blur on density map
    'colormap': 'plasma',    # Perceptually uniform colormap for visualization
    'export_json': True,     # Whether to export JSON metrics
    'export_images': True,   # Whether to export heat-map images
    'risk_score_clip': 0.95, # Percentile to clip risk scores to avoid outliers
    'log_format': 'ndjson',  # Format for structured logging
}

class NearMissEvent:
    """Represents a single near-miss or collision event."""
    
    def __init__(self, frame_idx: int, timestamp: float, 
                 p_center: Tuple[float, float], v_center: Tuple[float, float],
                 iou: float, centroid_distance: float, is_collision: bool):
        """
        Initialize near-miss event.
        
        Args:
            frame_idx: Frame index where event occurred
            timestamp: Timestamp in seconds
            p_center: Person center coordinates (x, y)
            v_center: Vehicle center coordinates (x, y)
            iou: Intersection over Union value
            centroid_distance: Distance between centroids
            is_collision: Whether this is a collision (red) or near-miss (yellow)
        """
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.p_center = p_center
        self.v_center = v_center
        
        # Interaction point is between person and vehicle (weighted toward person for safety)
        self.location = (
            0.7 * p_center[0] + 0.3 * v_center[0],
            0.7 * p_center[1] + 0.3 * v_center[1]
        )
        
        self.iou = iou
        self.centroid_distance = centroid_distance
        self.is_collision = is_collision
        
        # For time-of-day analytics
        self.hour = datetime.fromtimestamp(timestamp).hour if timestamp > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON export."""
        return {
            'frame': int(self.frame_idx),
            'timestamp': float(self.timestamp),
            'location': {
                'x': float(self.location[0]),
                'y': float(self.location[1])
            },
            'iou': float(self.iou),
            'centroid_distance': float(self.centroid_distance),
            'is_collision': bool(self.is_collision),
            'hour': int(self.hour)
        }


class HotspotCluster:
    """Represents a cluster of near-miss events in a specific location."""
    
    def __init__(self, cluster_id: int, events: List[NearMissEvent], 
                 center: Tuple[float, float], radius: float):
        """
        Initialize hotspot cluster.
        
        Args:
            cluster_id: Unique identifier for cluster
            events: List of events in this cluster
            center: Center point (x, y) of cluster 
            radius: Approximate radius of cluster
        """
        self.id = cluster_id
        self.events = events
        self.center = center
        self.radius = radius
        self.count = len(events)
        self.collision_count = sum(1 for e in events if e.is_collision)
        
        # Calculate cluster metrics
        if events:
            timestamps = [e.timestamp for e in events]
            self.start_time = min(timestamps)
            self.end_time = max(timestamps)
            self.duration = self.end_time - self.start_time
            
            # Calculate peak hour
            hours = [e.hour for e in events]
            hour_counts = defaultdict(int)
            for h in hours:
                hour_counts[h] += 1
            self.peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate avg metrics
            self.avg_iou = np.mean([e.iou for e in events])
            self.avg_dist = np.mean([e.centroid_distance for e in events])
            
            # Post Encroachment Time approximation (time between conflicts)
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps)
                diffs = [sorted_timestamps[i+1] - sorted_timestamps[i] 
                         for i in range(len(sorted_timestamps)-1)]
                self.avg_pet = np.mean(diffs) if diffs else 0
            else:
                self.avg_pet = 0
        else:
            self.start_time = 0
            self.end_time = 0
            self.duration = 0
            self.peak_hour = 0
            self.avg_iou = 0
            self.avg_dist = 0
            self.avg_pet = 0
        
        # Risk score (higher = more dangerous)
        # Enhanced formula with quadratic distance penalty
        # (event count × red ratio) ÷ (avg dist² × avg time between events)
        risk_numerator = self.count * (self.collision_count / max(1, self.count))
        # Quadratic term on proximity (1/d²) with guard against division by zero
        dist_factor = max(1.0, self.avg_dist**2)
        time_factor = max(0.1, self.avg_pet)
        risk_denominator = dist_factor * time_factor
        self.risk_score = 100 * risk_numerator / risk_denominator
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hotspot to dictionary for JSON export."""
        return {
            'id': int(self.id),
            'center': {
                'x': float(self.center[0]), 
                'y': float(self.center[1])
            },
            'radius': float(self.radius),
            'events': int(self.count),
            'collisions': int(self.collision_count),
            'near_misses': int(self.count - self.collision_count),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'duration': float(self.duration),
            'peak_hour': int(self.peak_hour),
            'risk_score': float(self.risk_score),
            'avg_metrics': {
                'iou': float(self.avg_iou),
                'distance': float(self.avg_dist),
                'pet': float(self.avg_pet)
            }
        }


class HeatmapAnalyzer:
    """Generates heat-maps and analytics from collision/near-miss event data."""
    
    def __init__(self, config: Dict[str, Any] = None, frame_shape: Tuple[int, int, int] = None, centroid_threshold: float = None):
        """
        Initialize heat-map analyzer.
        
        Args:
            config: Configuration options (defaults to HEATMAP_CONFIG)
            frame_shape: Shape of video frames (height, width, channels)
            centroid_threshold: Distance threshold for centroid-based detection (px)
        """
        self.config = config or HEATMAP_CONFIG.copy()
        self.events: List[NearMissEvent] = []
        self.hotspots: List[HotspotCluster] = []
        self.frame_shape = frame_shape
        self.output_dir = "heatmap_results"
        
        # Auto-compute DBSCAN epsilon based on centroid threshold if provided
        if centroid_threshold is not None:
            self.config['eps'] = centroid_threshold * 0.6
        elif self.config['eps'] is None and frame_shape is not None:
            # Default calculation based on frame dimensions
            height, width = frame_shape[:2]
            diagonal = np.sqrt(width**2 + height**2)
            self.config['eps'] = 0.09 * diagonal  # ~9% of diagonal
        elif self.config['eps'] is None:
            # Fallback default
            self.config['eps'] = 48
            
        # Create log directory for structured logs
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = int(time.time())
        self.log_file = os.path.join(self.log_dir, f"events_{timestamp}.ndjson")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_event(self, event: NearMissEvent) -> None:
        """Add a near-miss or collision event to the analyzer and log it."""
        self.events.append(event)
        
        # Write structured log in NDJSON format
        if self.config.get('log_format') == 'ndjson':
            log_entry = {
                'ts': time.time(),
                'event': 'collision' if event.is_collision else 'near_miss',
                'frame': event.frame_idx,
                'timestamp': event.timestamp,
                'iou': float(event.iou),
                'd': float(event.centroid_distance),
                'loc': {'x': float(event.location[0]), 'y': float(event.location[1])}
            }
            
            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def cluster_events(self) -> List[HotspotCluster]:
        """
        Cluster events using DBSCAN to identify hotspots.
        
        Returns:
            List of hotspot clusters
        """
        if not self.events or len(self.events) < self.config['min_events']:
            print(f"Not enough events for clustering. Need at least {self.config['min_events']}.")
            return []
        
        # Extract locations for clustering
        locations = np.array([e.location for e in self.events])
        
        # Run DBSCAN with resolution-aware epsilon
        db = DBSCAN(
            eps=self.config['eps'], 
            min_samples=self.config['min_samples']
        ).fit(locations)
        
        labels = db.labels_
        unique_labels = set(labels)
        
        print(f"DBSCAN clustering with eps={self.config['eps']:.1f}px found {len(unique_labels)-1} clusters")
        print(f"Noise points: {np.sum(labels == -1)} / {len(labels)}")
        
        # Organize events by cluster
        self.hotspots = []
        for cluster_id in unique_labels:
            # Skip noise points
            if cluster_id == -1:
                continue
                
            # Get all events in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_events = [self.events[i] for i in cluster_indices]
            
            if len(cluster_events) < self.config['min_events']:
                continue
                
            # Calculate cluster center and radius
            cluster_points = np.array([e.location for e in cluster_events])
            center = np.mean(cluster_points, axis=0)
            
            # Radius = maximum distance from center to any point
            distances = np.linalg.norm(cluster_points - center, axis=1)
            radius = np.max(distances)
            
            # Create hotspot
            hotspot = HotspotCluster(
                cluster_id=int(cluster_id),
                events=cluster_events,
                center=tuple(center),
                radius=float(radius)
            )
            
            self.hotspots.append(hotspot)
        
        # Sort hotspots by risk score (descending)
        self.hotspots.sort(key=lambda h: h.risk_score, reverse=True)
        
        # Apply percentile clipping to risk scores if we have enough hotspots
        if len(self.hotspots) >= 3 and self.config.get('risk_score_clip'):
            clip_percentile = self.config['risk_score_clip']
            risk_scores = np.array([h.risk_score for h in self.hotspots])
            max_score = np.percentile(risk_scores, clip_percentile * 100)
            
            # Clip risk scores to prevent outliers from dominating
            for hotspot in self.hotspots:
                hotspot.risk_score = min(hotspot.risk_score, max_score)
            
            # Re-sort after clipping
            self.hotspots.sort(key=lambda h: h.risk_score, reverse=True)
        
        return self.hotspots
    
    def generate_density_map(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a density heat-map from events.
        
        Args:
            frame: Optional background frame for overlay
        
        Returns:
            Heat-map image as numpy array
        """
        if not self.events:
            print("No events to generate heat-map from.")
            return None
            
        if frame is not None:
            height, width = frame.shape[:2]
        elif self.frame_shape:
            height, width = self.frame_shape[:2]
        else:
            # Default size if no frame provided
            width, height = 1280, 720
        
        # Create density map
        resolution = self.config['resolution']
        map_width = int(width * resolution)
        map_height = int(height * resolution)
        
        density = np.zeros((map_height, map_width), dtype=np.float32)
        
        # Add each event to density map
        for event in self.events:
            x, y = event.location
            x = int(x * resolution)
            y = int(y * resolution)
            
            # Skip if outside bounds
            if x < 0 or x >= map_width or y < 0 or y >= map_height:
                continue
                
            # Higher weight for collision events
            weight = 2.0 if event.is_collision else 1.0
            density[y, x] += weight
        
        # Apply Gaussian blur to create smooth heat-map
        sigma = self.config['gaussian_sigma']
        density = cv2.GaussianBlur(density, (0, 0), sigma)
        
        # Normalize
        if np.max(density) > 0:
            density = density / np.max(density)
        
        return density
    
    def overlay_heatmap(self, frame: np.ndarray, density: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Overlay heat-map on a video frame.
        
        Args:
            frame: Background video frame
            density: Optional pre-computed density map
        
        Returns:
            Frame with heat-map overlay
        """
        if density is None:
            density = self.generate_density_map(frame)
            
        if density is None:
            return frame
        
        # Resize density map to match frame if needed
        height, width = frame.shape[:2]
        if density.shape[:2] != (height, width):
            density = cv2.resize(density, (width, height))
        
        # Create colored heat-map using selected colormap
        cmap = cm.get_cmap(self.config['colormap'])
        colored_heatmap = cmap(density)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # Create alpha mask based on intensity
        alpha = self.config['alpha'] * density
        alpha = np.expand_dims(alpha, axis=2)
        
        # Ensure alpha has the correct shape for broadcasting
        if alpha.shape[2] != 1:
            alpha = alpha[:, :, :1]
        
        # Blend with original frame
        heat_overlay = frame.copy()
        for c in range(3):
            heat_overlay[:, :, c] = frame[:, :, c] * (1 - alpha[:, :, 0]) + colored_heatmap[:, :, c] * alpha[:, :, 0]
            
        return heat_overlay.astype(np.uint8)
    
    def draw_hotspots(self, frame: np.ndarray, draw_labels: bool = True) -> np.ndarray:
        """
        Draw hotspot circles and labels on a frame.
        
        Args:
            frame: Video frame to draw on
            draw_labels: Whether to draw labels with metrics
        
        Returns:
            Frame with hotspot visualization
        """
        result = frame.copy()
        
        # Make sure we have hotspots
        if not self.hotspots:
            self.cluster_events()
            
        if not self.hotspots:
            return result
            
        # Draw each hotspot
        for i, hotspot in enumerate(self.hotspots):
            # Normalize risk score for color
            risk_normalized = min(1.0, hotspot.risk_score / 100)
            
            # Color based on risk (green to red)
            color = tuple([int(c*255) for c in plt.cm.RdYlGn_r(risk_normalized)[:3]])
            
            # Draw circle
            cv2.circle(
                result,
                (int(hotspot.center[0]), int(hotspot.center[1])),
                int(hotspot.radius),
                color,
                2
            )
            
            # Draw label
            if draw_labels:
                label = f"Risk:{hotspot.risk_score:.1f} Events:{hotspot.count}"
                cv2.putText(
                    result,
                    label,
                    (int(hotspot.center[0]) - 60, int(hotspot.center[1]) - int(hotspot.radius) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
        
        return result
    
    def export_results(self, background_frame: Optional[np.ndarray] = None) -> Dict[str, str]:
        """
        Export analytics results to files (JSON data, static images).
        
        Args:
            background_frame: Optional background frame for visualization
        
        Returns:
            Dictionary with paths to generated files
        """
        timestamp = int(time.time())
        result_paths = {}
        
        # Make sure we have hotspots
        if not self.hotspots:
            self.cluster_events()
        
        # 1. Export JSON data
        if self.config['export_json']:
            # Make sure config is JSON serializable
            safe_config = {}
            for k, v in self.config.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_config[k] = v
                else:
                    safe_config[k] = str(v)  # Convert non-serializable types to string
            
            json_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'event_count': len(self.events),
                    'hotspot_count': len(self.hotspots),
                    'config': safe_config
                },
                'hotspots': [h.to_dict() for h in self.hotspots],
                'events': [e.to_dict() for e in self.events]
            }
            
            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            json_path = os.path.join(self.output_dir, f"heatmap_data_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, cls=NumpyEncoder)
            
            result_paths['json'] = json_path
            print(f"Exported JSON data to {json_path}")
        
        # 2. Export visualizations
        if self.config['export_images'] and (background_frame is not None or self.frame_shape):
            # Generate or use background
            if background_frame is None:
                height, width = self.frame_shape[:2]
                background = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                background = background_frame.copy()
            
            # Generate and save raw density map
            density = self.generate_density_map(background)
            if density is not None:
                # Save density image
                density_colored = plt.cm.get_cmap(self.config['colormap'])(density)
                density_colored = (density_colored[:, :, :3] * 255).astype(np.uint8)
                
                density_path = os.path.join(self.output_dir, f"density_map_{timestamp}.png")
                cv2.imwrite(density_path, cv2.cvtColor(density_colored, cv2.COLOR_RGB2BGR))
                result_paths['density'] = density_path
                
                # Save overlay image
                overlay = self.overlay_heatmap(background, density)
                overlay_path = os.path.join(self.output_dir, f"heatmap_overlay_{timestamp}.png")
                cv2.imwrite(overlay_path, overlay)
                result_paths['overlay'] = overlay_path
                
                # Save hotspot visualization
                hotspot_viz = self.draw_hotspots(overlay, draw_labels=True)
                hotspot_path = os.path.join(self.output_dir, f"hotspots_{timestamp}.png")
                cv2.imwrite(hotspot_path, hotspot_viz)
                result_paths['hotspots'] = hotspot_path
                
                print(f"Exported visualization images to {self.output_dir}")
        
        return result_paths
    
    def print_analytics_summary(self) -> None:
        """Print a summary of hotspot analytics to console."""
        if not self.hotspots:
            self.cluster_events()
            
        if not self.hotspots:
            print("No hotspots identified. Try adjusting clustering parameters.")
            return
            
        print(f"\n===== NEAR-MISS HOTSPOT SUMMARY =====")
        print(f"Total events analyzed: {len(self.events)}")
        print(f"Hotspots identified: {len(self.hotspots)}")
        print(f"\nTOP RISK AREAS:")
        
        for i, hotspot in enumerate(self.hotspots[:5]):
            print(f"\n{i+1}. Risk Score: {hotspot.risk_score:.1f}")
            print(f"   Events: {hotspot.count} ({hotspot.collision_count} collisions, "
                  f"{hotspot.count - hotspot.collision_count} near-misses)")
            print(f"   Peak Hour: {hotspot.peak_hour:02d}:00")
            print(f"   Avg Time Between Events: {hotspot.avg_pet:.2f}s")
            print(f"   Avg Distance: {hotspot.avg_dist:.2f}px")
            
        # Overall statistics
        all_events = [e for h in self.hotspots for e in h.events]
        if all_events:
            hours = [e.hour for e in all_events]
            hour_counts = defaultdict(int)
            for h in hours:
                hour_counts[h] += 1
                
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])
            
            print(f"\nOVERALL STATISTICS:")
            print(f"Peak activity hour: {peak_hour[0]:02d}:00 ({peak_hour[1]} events)")
            print(f"Total area covered by hotspots: "
                  f"{sum(np.pi * h.radius**2 for h in self.hotspots):.1f} px²")
            
        print(f"\nFor detailed metrics, check the JSON export.")


# Helper function to convert box coordinates to center point
def get_center_point(box):
    """Calculate center point from bounding box coordinates (x1, y1, x2, y2)."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# Command-line execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate heat-map analytics from collision events")
    parser.add_argument("--events-json", type=str, help="Path to events JSON file")
    parser.add_argument("--background", type=str, help="Path to background frame for visualization")
    parser.add_argument("--eps", type=float, default=30, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples parameter")
    
    args = parser.parse_args()
    
    # Custom config
    config = HEATMAP_CONFIG.copy()
    config['eps'] = args.eps
    config['min_samples'] = args.min_samples
    
    # Create analyzer
    analyzer = HeatmapAnalyzer(config)
    
    # Load events from JSON if provided
    if args.events_json and os.path.exists(args.events_json):
        with open(args.events_json, 'r') as f:
            data = json.load(f)
            
        if 'events' in data:
            for event_data in data['events']:
                x = event_data['location']['x']
                y = event_data['location']['y']
                
                event = NearMissEvent(
                    frame_idx=event_data['frame'],
                    timestamp=event_data['timestamp'],
                    p_center=(x-10, y-10),  # Dummy values offset from location
                    v_center=(x+10, y+10),  # Dummy values offset from location
                    iou=event_data.get('iou', 0.0),
                    centroid_distance=event_data.get('centroid_distance', 0.0),
                    is_collision=event_data.get('is_collision', False)
                )
                analyzer.add_event(event)
    
    # Load background image if provided
    background = None
    if args.background and os.path.exists(args.background):
        background = cv2.imread(args.background)
    
    # Generate results
    analyzer.cluster_events()
    analyzer.print_analytics_summary()
    analyzer.export_results(background)
