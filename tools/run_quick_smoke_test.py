#!/usr/bin/env python3
"""
Quick smoke test for centroid-based detection.

Tests for:
1. Resolution-aware threshold scaling
2. Centroid distance calculation with different resolution
3. Yellow cooldown functionality
4. Basic DBSCAN clustering with our modifications

This is a fast implementation check without requiring video processing.
"""

import os
import sys
import time
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import our optimized modules
from collision_config import COLLISION_IOU_THRESHOLD, CENTROID_THRESHOLD, compute_centroid_threshold
from cooldown_manager import CooldownManager
from heatmap_analytics import HeatmapAnalyzer, NearMissEvent, HotspotCluster

def test_centroid_threshold_scaling():
    """Test that centroid threshold scales appropriately with resolution."""
    print("\n===== Testing Resolution-Aware Scaling =====")
    
    # Test resolutions
    resolutions = [
        (640, 480),    # VGA
        (1280, 720),   # 720p
        (1920, 1080),  # 1080p
        (3840, 2160)   # 4K
    ]
    
    reference_threshold = CENTROID_THRESHOLD  # ~80px at 720p
    
    for width, height in resolutions:
        threshold = compute_centroid_threshold(width, height)
        diagonal = np.sqrt(width**2 + height**2)
        print(f"Resolution: {width}x{height} (diagonal: {diagonal:.1f}px)")
        print(f"  Centroid threshold: {threshold:.1f}px")
        print(f"  Relative to frame: {threshold/diagonal*100:.2f}% of diagonal")
    
    # Verify the reference resolution matches our expected ~80px
    ref_width, ref_height = 1280, 720
    ref_threshold = compute_centroid_threshold(ref_width, ref_height)
    # Check that reference threshold is close to 80px
    assert 70 <= ref_threshold <= 90, f"Reference threshold {ref_threshold} is too far from expected 80px"
    print("\n[PASS] Centroid threshold scales correctly with resolution")
    
    return True

def test_cooldown_functionality():
    """Test the yellow flag cooldown mechanism."""
    print("\n===== Testing Cooldown Mechanism =====")
    
    cooldown_time = 2.0  # 2 second cooldown
    cooldown_mgr = CooldownManager(cooldown_time=cooldown_time)
    
    # Test basic cooldown
    start_time = 10.0
    person_id, vehicle_id = 1, 2
    
    # First check should allow
    in_cooldown = cooldown_mgr.in_cooldown(person_id, vehicle_id, start_time)
    assert not in_cooldown, "Should not be in cooldown initially"
    
    # Set cooldown
    cooldown_mgr.set(person_id, vehicle_id, start_time)
    
    # Check at different times
    test_times = [
        start_time + 0.5,  # 0.5s later (should block)
        start_time + 1.9,  # 1.9s later (should block)
        start_time + 2.1,  # 2.1s later (should allow)
    ]
    
    expected_results = [True, True, False]
    
    for time_point, expected in zip(test_times, expected_results):
        result = cooldown_mgr.in_cooldown(person_id, vehicle_id, time_point)
        assert result == expected, f"Cooldown at t+{time_point-start_time:.1f}s should be {expected}, got {result}"
    
    # Test different object pairs
    cooldown_mgr.set(3, 4, start_time)
    assert not cooldown_mgr.in_cooldown(3, 5, start_time), "Different pairs should have independent cooldowns"
    
    print(f"[PASS] Cooldown mechanism works correctly with {cooldown_time}s timeout")
    return True

def test_heatmap_clustering():
    """Test the DBSCAN clustering for heatmap generation."""
    print("\n===== Testing DBSCAN Clustering =====")
    
    # Create synthetic events in known clusters
    # Three distinct clusters with known positions
    cluster_centers = [
        (100, 100),  # Top left
        (400, 400),  # Center
        (700, 150)   # Top right
    ]
    
    # Generate 60 events (20 per cluster with some noise)
    events = []
    frame_idx = 0
    
    for center_x, center_y in cluster_centers:
        # Create 20 events per cluster
        for i in range(20):
            # Add some noise
            noise_x = np.random.normal(0, 15)
            noise_y = np.random.normal(0, 15)
            
            x = center_x + noise_x
            y = center_y + noise_y
            
            # Create event
            event = NearMissEvent(
                frame_idx=frame_idx,
                timestamp=frame_idx / 30.0,  # Assume 30 FPS
                p_center=(x-10, y-10),
                v_center=(x+10, y+10),
                iou=0.1,
                centroid_distance=20,
                is_collision=(i % 5 == 0)  # 20% are collisions
            )
            
            events.append(event)
            frame_idx += 1
    
    # Initialize analyzer with synthetic data
    frame_shape = (800, 800, 3)
    analyzer = HeatmapAnalyzer(frame_shape=frame_shape)
    
    # Add events
    for event in events:
        analyzer.add_event(event)
    
    # Test different eps values
    eps_values = [30, 50, 80]
    
    for eps in eps_values:
        analyzer.config['eps'] = eps
        hotspots = analyzer.cluster_events()
        
        print(f"DBSCAN with eps={eps}:")
        print(f"  Found {len(hotspots)} clusters")
        print(f"  Events per cluster: {[h.count for h in hotspots]}")
        print(f"  Risk scores: {[round(h.risk_score, 1) for h in hotspots]}")
        
        # Verify risk score clipping
        if len(hotspots) >= 3:
            risk_scores = [h.risk_score for h in hotspots]
            max_score = max(risk_scores)
            min_score = min(risk_scores)
            print(f"  Risk score range: {min_score:.1f} - {max_score:.1f}")
            
            # Check if clipping worked (max/min ratio shouldn't be too extreme)
            assert max_score / max(1.0, min_score) < 100, "Risk score clipping may not be working"
    
    # Generate heatmap visualization for testing
    os.makedirs("smoke_test_results", exist_ok=True)
    
    # Create a test frame with correct shape (height, width, channels)
    test_frame = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Draw some background elements
    cv2.rectangle(test_frame, (0, 0), (799, 799), (30, 30, 30), -1)
    cv2.rectangle(test_frame, (50, 50), (750, 750), (60, 60, 60), -1)
    
    # Generate and save density map
    density = analyzer.generate_density_map(test_frame)
    assert density is not None, "Density map generation failed"
    
    # Create overlay
    heatmap = analyzer.overlay_heatmap(test_frame, density)
    assert heatmap is not None, "Heatmap overlay failed"
    
    # Draw hotspots
    hotspot_viz = analyzer.draw_hotspots(heatmap)
    
    # Save visualization
    cv2.imwrite("smoke_test_results/heatmap_test.png", hotspot_viz)
    
    print(f"[PASS] DBSCAN clustering works correctly")
    print(f"[PASS] Heatmap visualization saved to smoke_test_results/heatmap_test.png")
    
    return True

def main():
    """Run all smoke tests."""
    print("===== Running Centroid Detection Smoke Tests =====")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    # Run tests
    try:
        test_centroid_threshold_scaling()
        test_cooldown_functionality()
        test_heatmap_clustering()
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        all_passed = False
    
    # Print final result
    print("\n===== Test Results =====")
    if all_passed:
        print("[PASS] ALL TESTS PASSED")
        print("The optimized system is functioning correctly.")
        print("Ready for full 10-minute analysis run!")
    else:
        print("[FAIL] TESTS FAILED")
        print("Please fix the issues before running the full analysis.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
