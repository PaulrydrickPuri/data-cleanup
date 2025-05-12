#!/usr/bin/env python3
"""
Analytics visualization dashboard for collision detection results.

This script provides visualization of:
1. Heat-maps of collision hot spots
2. Near-miss event distributions
3. Risk metrics and analytics
4. Comparison of detection methods
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import json
import glob
from pathlib import Path
from datetime import datetime

def show_training_results(results_dir):
    """Display the training results from the YOLOv8 output directory."""
    results_dir = Path(results_dir)
    
    # Check if the results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Get the results.png path
    results_img_path = results_dir / "results.png"
    if not results_img_path.exists():
        print(f"Error: Results image not found at {results_img_path}")
        return
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Plot the main results
    ax1 = fig.add_subplot(221)
    img = mpimg.imread(results_img_path)
    ax1.imshow(img)
    ax1.set_title("Training Metrics (loss, precision, recall, mAP)")
    ax1.axis('off')
    
    # Plot confusion matrix if available
    confusion_img_path = results_dir / "confusion_matrix_normalized.png"
    if confusion_img_path.exists():
        ax2 = fig.add_subplot(222)
        img = mpimg.imread(confusion_img_path)
        ax2.imshow(img)
        ax2.set_title("Normalized Confusion Matrix")
        ax2.axis('off')
    
    # Plot PR curve if available
    pr_img_path = results_dir / "PR_curve.png"
    if pr_img_path.exists():
        ax3 = fig.add_subplot(223)
        img = mpimg.imread(pr_img_path)
        ax3.imshow(img)
        ax3.set_title("Precision-Recall Curve")
        ax3.axis('off')
    
    # Plot F1 curve if available
    f1_img_path = results_dir / "F1_curve.png"
    if f1_img_path.exists():
        ax4 = fig.add_subplot(224)
        img = mpimg.imread(f1_img_path)
        ax4.imshow(img)
        ax4.set_title("F1 Score Curve")
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show prediction examples if available
    val_img_paths = [
        results_dir / "val_batch0_pred.jpg",
        results_dir / "val_batch1_pred.jpg",
        results_dir / "val_batch2_pred.jpg"
    ]
    
    if any(p.exists() for p in val_img_paths):
        fig2 = plt.figure(figsize=(15, 10))
        
        for i, img_path in enumerate([p for p in val_img_paths if p.exists()]):
            ax = fig2.add_subplot(1, len(val_img_paths), i+1)
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"Validation Batch {i} Predictions")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def show_collision_analytics(results_dir):
    """Display the collision analytics results from the centroid detection."""
    results_dir = Path(results_dir)
    
    # Check if the results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Look for heatmap data JSON files
    json_files = list(results_dir.glob("**/heatmap_data_*.json"))
    if not json_files:
        print(f"Error: No heat-map data files found in {results_dir}")
        return
    
    # Use the most recent JSON file (highest timestamp)
    json_file = sorted(json_files, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    print(f"Using analytics data from: {json_file}")
    
    # Load the JSON data
    with open(json_file) as f:
        analytics_data = json.load(f)
    
    # Look for heatmap visualization images
    heatmap_files = list(results_dir.glob("**/heatmap_*.png"))
    background_heatmap = next((f for f in heatmap_files if "background" in f.name), None)
    cluster_heatmap = next((f for f in heatmap_files if "clusters" in f.name), None)
    
    # Get segment metrics if available
    metrics_dashboard = list(results_dir.glob("**/metrics_dashboard.png"))
    if metrics_dashboard:
        metrics_dashboard = metrics_dashboard[0]
        
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Collision Detection Analytics Dashboard", fontsize=16)
    
    # Plot the heatmap visualization
    ax1 = fig.add_subplot(221)
    if background_heatmap and background_heatmap.exists():
        img = mpimg.imread(background_heatmap)
        ax1.imshow(img)
        ax1.set_title("Collision Heat-Map")
    else:
        ax1.text(0.5, 0.5, "Heat-map image not available", 
                 horizontalalignment='center', verticalalignment='center')
    ax1.axis('off')
    
    # Plot cluster visualization if available
    ax2 = fig.add_subplot(222)
    if cluster_heatmap and cluster_heatmap.exists():
        img = mpimg.imread(cluster_heatmap)
        ax2.imshow(img)
        ax2.set_title("DBSCAN Cluster Analysis")
    else:
        ax2.text(0.5, 0.5, "Cluster visualization not available", 
                 horizontalalignment='center', verticalalignment='center')
    ax2.axis('off')
    
    # Plot event timeline or metrics dashboard if available
    ax3 = fig.add_subplot(223)
    if metrics_dashboard and metrics_dashboard.exists():
        img = mpimg.imread(metrics_dashboard)
        ax3.imshow(img)
        ax3.set_title("Metrics Dashboard")
        ax3.axis('off')
    else:
        # Create a simple event timeline from JSON data
        events = analytics_data.get('events', [])
        if events:
            # Extract collision vs near-miss events by frame number
            frames = [e['frame'] for e in events]
            is_collision = [1 if e['is_collision'] else 0 for e in events]
            ax3.stem(frames, is_collision, linefmt='r-', markerfmt='ro', basefmt='k-')
            ax3.set_title("Collision Event Timeline")
            ax3.set_xlabel("Frame Number")
            ax3.set_ylabel("Collision (1) / Near-Miss (0)")
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Near-Miss', 'Collision'])
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No event data available", 
                     horizontalalignment='center', verticalalignment='center')
    
    # Plot hotspot summary
    ax4 = fig.add_subplot(224)
    hotspots = analytics_data.get('hotspots', [])
    if hotspots:
        risk_scores = [h['risk_score'] for h in hotspots]
        event_counts = [h['events'] for h in hotspots]
        collision_counts = [h['collisions'] for h in hotspots]
        near_miss_counts = [h['near_misses'] for h in hotspots]
        
        # Limit to top 5 hotspots by risk score
        if len(hotspots) > 5:
            # Sort by risk score
            indices = np.argsort(risk_scores)[-5:]
            risk_scores = [risk_scores[i] for i in indices]
            event_counts = [event_counts[i] for i in indices]
            collision_counts = [collision_counts[i] for i in indices]
            near_miss_counts = [near_miss_counts[i] for i in indices]
            
        # Plot stacked bar chart of collisions vs near-misses
        hotspot_ids = [f"H{i+1}" for i in range(len(risk_scores))]
        ax4.bar(hotspot_ids, collision_counts, label='Collisions', color='darkred')
        ax4.bar(hotspot_ids, near_miss_counts, bottom=collision_counts, label='Near-Misses', color='orange')
        
        # Overlay risk scores as text
        for i, (h, risk) in enumerate(zip(hotspot_ids, risk_scores)):
            total = collision_counts[i] + near_miss_counts[i]
            ax4.text(i, total + 0.1, f"Risk: {risk:.1f}", ha='center')
        
        ax4.set_title("Hotspot Analysis")
        ax4.set_xlabel("Hotspot ID")
        ax4.set_ylabel("Event Count")
        ax4.legend()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No hotspot data available", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Add metadata
    metadata = analytics_data.get('metadata', {})
    timestamp = metadata.get('timestamp', None)
    if timestamp:
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.02, 0.02, f"Generated: {date_str}", fontsize=8)
    
    plt.figtext(0.98, 0.02, f"Total Events: {len(analytics_data.get('events', []))}", 
                 fontsize=8, horizontalalignment='right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Display collision detection analytics")
    parser.add_argument("--results_dir", type=str, default="centroid_analytics_results",
                        help="Path to collision analytics results directory")
    parser.add_argument("--segment", type=str, default=None,
                        help="Specific segment to display (e.g., 'segment_3')")
    parser.add_argument("--mode", type=str, choices=["yolo", "collision"], default="collision",
                        help="Type of results to display")
    args = parser.parse_args()
    
    if args.mode == "yolo":
        show_training_results(args.results_dir)
    else:
        # If a specific segment is provided, look in that subdirectory
        if args.segment:
            results_path = os.path.join(args.results_dir, args.segment)
        else:
            results_path = args.results_dir
        
        show_collision_analytics(results_path)

if __name__ == "__main__":
    main()
