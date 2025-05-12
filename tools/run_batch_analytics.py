#!/usr/bin/env python3
"""
Batch analytics processor for all gold segments.

This script processes all gold segments to generate:
1. Individual segment analytics
2. Combined heat-map of all near-miss events 
3. Consolidated risk metrics dashboard export

Uses the optimized centroid-based detection with:
- Resolution-aware proximity thresholds
- 2-second yellow flag cooldown
- DBSCAN clustering with perceptually uniform colormaps
"""

import os
import sys
import time
import subprocess
import glob
import json
import logging
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_analytics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_analytics')

def find_segments(directory_path, pattern="*_segment_*_output.mp4"):
    """Find all gold segments in the given directory."""
    segments = []
    paths = glob.glob(os.path.join(directory_path, pattern))
    
    # Group by segment number
    segment_groups = {}
    for path in paths:
        filename = os.path.basename(path)
        
        # Extract variant and segment number
        # Pattern is like: centroid_segment_3_output.mp4
        if "segment_" in filename and filename.endswith("_output.mp4"):
            parts = filename.split('_')
            variant = parts[0]
            # Segment number is typically the part after 'segment_'
            segment_idx = parts.index('segment')
            if segment_idx + 1 < len(parts):
                try:
                    segment_num = int(parts[segment_idx + 1])
                    
                    if segment_num not in segment_groups:
                        segment_groups[segment_num] = []
                    
                    segment_groups[segment_num].append({
                        'path': path,
                        'variant': variant, 
                        'number': segment_num
                    })
                except ValueError:
                    # Skip if segment number is not an integer
                    pass
    
    # Get the centroid variants for each segment number
    for segment_num, variants in segment_groups.items():
        for v in variants:
            if v['variant'] == 'centroid':
                segments.append(v)
                break
        else:
            # If no centroid variant, use any variant
            if variants:
                segments.append(variants[0])
    
    # Sort by segment number
    segments.sort(key=lambda x: x['number'])
    return segments

def process_segment(segment_info, output_dir, visualize=False):
    """Process a single segment using the centroid analytics script."""
    segment_path = segment_info['path']
    segment_num = segment_info['number']
    
    # Create segment-specific output directory
    segment_output = os.path.join(output_dir, f"segment_{segment_num}")
    os.makedirs(segment_output, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "tools/run_centroid_analytics.py",
        "--video", segment_path,
        "--output", segment_output,
        "--segment-name", f"segment_{segment_num}"
    ]
    
    if visualize:
        cmd.append("--visualize")
    
    # Run command
    logger.info(f"Processing segment {segment_num}: {os.path.basename(segment_path)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Segment {segment_num} processed successfully in {time.time() - start_time:.1f}s")
        return {
            'segment': segment_num,
            'output_dir': segment_output,
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'time': time.time() - start_time
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing segment {segment_num}: {e}")
        return {
            'segment': segment_num,
            'output_dir': segment_output,
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'time': time.time() - start_time
        }

def collect_events(results):
    """Collect all events from processed segments."""
    all_events = []
    all_hotspots = []
    
    for result in results:
        if not result['success']:
            continue
            
        # Find the JSON output file
        json_files = glob.glob(os.path.join(result['output_dir'], "heatmap_data_*.json"))
        if not json_files:
            logger.warning(f"No JSON data found for segment {result['segment']}")
            continue
        
        # Use the most recent JSON file
        json_file = max(json_files, key=os.path.getmtime)
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Add segment information to events
            segment_events = data.get('events', [])
            for event in segment_events:
                event['segment'] = result['segment']
            
            all_events.extend(segment_events)
            
            # Add segment information to hotspots
            segment_hotspots = data.get('hotspots', [])
            for hotspot in segment_hotspots:
                hotspot['segment'] = result['segment']
            
            all_hotspots.extend(segment_hotspots)
            
            logger.info(f"Added {len(segment_events)} events and {len(segment_hotspots)} hotspots from segment {result['segment']}")
        except Exception as e:
            logger.error(f"Error reading JSON data for segment {result['segment']}: {e}")
    
    return all_events, all_hotspots

def generate_combined_heatmap(events, hotspots, output_dir, frame_shape=(720, 1280, 3)):
    """Generate a combined heat-map from all events."""
    if not events:
        logger.warning("No events to generate combined heat-map")
        return False
    
    # Create background frame
    background = np.zeros(frame_shape, dtype=np.uint8)
    background[:] = (30, 30, 30)  # Dark gray background
    
    # Create density map (resolution 2x for better detail)
    density_height, density_width = frame_shape[:2]
    density = np.zeros((density_height, density_width), dtype=np.float32)
    
    # Add all events to density map
    for event in events:
        x = int(event['location']['x'])
        y = int(event['location']['y'])
        
        # Skip if outside bounds
        if x < 0 or x >= density_width or y < 0 or y >= density_height:
            continue
        
        # Higher weight for collision events
        weight = 2.0 if event['is_collision'] else 1.0
        
        # Add to density with a small radius for better visibility
        cv2.circle(density, (x, y), 5, weight, -1)
    
    # Apply Gaussian blur to create smooth heat-map
    density = cv2.GaussianBlur(density, (0, 0), 25)
    
    # Normalize
    if np.max(density) > 0:
        density = density / np.max(density)
    
    # Create colored heat-map using plasma colormap (perceptually uniform)
    plasma_cmap = plt.cm.get_cmap('plasma')
    heatmap_colored = plasma_cmap(density)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Create alpha mask based on intensity
    alpha = 0.7 * density
    alpha = np.expand_dims(alpha, axis=2)
    
    # Blend with background
    heatmap_overlay = background.copy()
    for c in range(3):
        heatmap_overlay[:, :, c] = background[:, :, c] * (1 - alpha[:, :, 0]) + heatmap_colored[:, :, c] * alpha[:, :, 0]
    
    # Draw hotspot circles
    for hotspot in hotspots:
        # Normalize risk score for color
        risk_normalized = min(1.0, hotspot['risk_score'] / 100)
        
        # Create a perceptually meaningful color (green to red)
        color = tuple([int(c*255) for c in plt.cm.RdYlGn_r(risk_normalized)[:3]])
        
        # Draw circle
        cv2.circle(
            heatmap_overlay,
            (int(hotspot['center']['x']), int(hotspot['center']['y'])),
            int(hotspot['radius']),
            color,
            2
        )
        
        # Add label
        label = f"Risk:{hotspot['risk_score']:.1f} (S{hotspot['segment']})"
        cv2.putText(
            heatmap_overlay,
            label,
            (int(hotspot['center']['x']) - 60, int(hotspot['center']['y']) - int(hotspot['radius']) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    # Add title and legend
    cv2.putText(
        heatmap_overlay,
        "Combined Near-Miss Heat-Map",
        (frame_shape[1]//2 - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    
    # Add event counts
    collision_count = sum(1 for e in events if e['is_collision'])
    potential_count = len(events) - collision_count
    
    cv2.putText(
        heatmap_overlay,
        f"Total Events: {len(events)} ({collision_count} collisions, {potential_count} near-misses)",
        (20, frame_shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Save results
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Save combined heat-map
    heatmap_path = os.path.join(combined_dir, "combined_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_overlay)
    
    # Save raw density map
    density_colored = (plasma_cmap(density)[:, :, :3] * 255).astype(np.uint8)
    density_path = os.path.join(combined_dir, "combined_density.png")
    cv2.imwrite(density_path, density_colored)
    
    # Save combined data as JSON
    timestamp = int(time.time())
    
    combined_data = {
        'timestamp': timestamp,
        'total_events': len(events),
        'total_collisions': collision_count,
        'total_near_misses': potential_count,
        'hotspots': len(hotspots),
        'events': events,
        'hotspots': hotspots
    }
    
    json_path = os.path.join(combined_dir, f"combined_analytics_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    logger.info(f"Combined heat-map saved to {heatmap_path}")
    logger.info(f"Combined analytics saved to {json_path}")
    
    return heatmap_path, json_path

def generate_metrics_dashboard(events, hotspots, output_dir):
    """Generate a metrics dashboard visualization of the near-miss analytics."""
    if not events or not hotspots:
        logger.warning("Insufficient data for metrics dashboard")
        return False
    
    # Create figure with multiple subplots
    plt.figure(figsize=(14, 10))
    
    # 1. Event counts by segment
    segment_counts = {}
    for event in events:
        segment = event['segment']
        if segment not in segment_counts:
            segment_counts[segment] = {'collisions': 0, 'near_misses': 0}
        
        if event['is_collision']:
            segment_counts[segment]['collisions'] += 1
        else:
            segment_counts[segment]['near_misses'] += 1
    
    # Sort segments
    segments = sorted(segment_counts.keys())
    collision_counts = [segment_counts[s]['collisions'] for s in segments]
    near_miss_counts = [segment_counts[s]['near_misses'] for s in segments]
    
    # Plot event counts
    plt.subplot(2, 2, 1)
    bar_width = 0.35
    x = np.arange(len(segments))
    plt.bar(x - bar_width/2, collision_counts, bar_width, label='Collisions', color='red')
    plt.bar(x + bar_width/2, near_miss_counts, bar_width, label='Near-Misses', color='orange')
    plt.xlabel('Segment')
    plt.ylabel('Count')
    plt.title('Events by Segment')
    plt.xticks(x, [f"S{s}" for s in segments])
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Risk scores by hotspot
    plt.subplot(2, 2, 2)
    hotspot_risks = [(h['id'], h['risk_score'], h['segment']) for h in hotspots]
    hotspot_risks.sort(key=lambda x: x[1], reverse=True)
    
    hotspot_ids = [f"H{h[0]}(S{h[2]})" for h in hotspot_risks[:10]]  # Top 10 hotspots
    risk_scores = [h[1] for h in hotspot_risks[:10]]
    
    plt.barh(hotspot_ids, risk_scores, color='darkred')
    plt.xlabel('Risk Score')
    plt.title('Top Risk Hotspots')
    plt.grid(alpha=0.3)
    
    # 3. Distance distribution
    plt.subplot(2, 2, 3)
    distances = [e['centroid_distance'] for e in events]
    
    # Create histogram  
    plt.hist(distances, bins=20, alpha=0.7, color='green')
    plt.axvline(x=80, color='r', linestyle='--', label='Threshold (80px)')
    plt.xlabel('Centroid Distance (px)')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. IoU distribution
    plt.subplot(2, 2, 4)
    ious = [e['iou'] for e in events]
    
    # Create histogram
    plt.hist(ious, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    plt.xlabel('IoU Value')
    plt.ylabel('Frequency')
    plt.title('IoU Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add title and adjust layout
    plt.suptitle('Near-Miss Analytics Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save dashboard
    dashboard_dir = os.path.join(output_dir, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    dashboard_path = os.path.join(dashboard_dir, "metrics_dashboard.png")
    plt.savefig(dashboard_path, dpi=150)
    
    logger.info(f"Metrics dashboard saved to {dashboard_path}")
    return dashboard_path

def main():
    """Main function to batch process all segments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch process gold segments for analytics")
    parser.add_argument("--input-dir", type=str, default="evaluation_results", 
                      help="Directory containing segment videos")
    parser.add_argument("--output-dir", type=str, default="centroid_analytics_results", 
                      help="Output directory for analytics")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization videos")
    parser.add_argument("--skip-existing", action="store_true", help="Skip segments with existing results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all segments
    segments = find_segments(args.input_dir)
    logger.info(f"Found {len(segments)} segments to process")
    
    if not segments:
        logger.error(f"No segments found in {args.input_dir}")
        return 1
    
    # Process each segment
    results = []
    start_time = time.time()
    
    for segment in segments:
        # Check if we should skip this segment
        if args.skip_existing:
            segment_dir = os.path.join(args.output_dir, f"segment_{segment['number']}")
            if os.path.exists(segment_dir) and os.listdir(segment_dir):
                logger.info(f"Skipping segment {segment['number']}: results already exist")
                
                # Create a placeholder result
                results.append({
                    'segment': segment['number'],
                    'output_dir': segment_dir,
                    'success': True,
                    'stdout': "",
                    'stderr': "",
                    'time': 0
                })
                continue
        
        # Process segment
        result = process_segment(segment, args.output_dir, args.visualize)
        results.append(result)
    
    # Collect all events and hotspots
    logger.info("Collecting events from all segments...")
    all_events, all_hotspots = collect_events(results)
    logger.info(f"Collected {len(all_events)} events and {len(all_hotspots)} hotspots in total")
    
    # Generate combined heat-map
    if all_events:
        logger.info("Generating combined heat-map...")
        heatmap_path, json_path = generate_combined_heatmap(all_events, all_hotspots, args.output_dir)
        
        # Generate metrics dashboard
        logger.info("Generating metrics dashboard...")
        dashboard_path = generate_metrics_dashboard(all_events, all_hotspots, args.output_dir)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    total_time = time.time() - start_time
    
    logger.info("\n===== Batch Processing Complete =====")
    logger.info(f"Processed {len(segments)} segments in {total_time:.1f} seconds")
    logger.info(f"Success rate: {success_count}/{len(segments)}")
    logger.info(f"Total events: {len(all_events)}")
    logger.info(f"Total hotspots: {len(all_hotspots)}")
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
