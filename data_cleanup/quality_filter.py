#!/usr/bin/env python3
"""
Quality filter for images and object crops based on blur, size, and exposure.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityFilter:
    """Filter images and object crops based on quality metrics."""
    
    def __init__(self, 
                 blur_threshold: float = 100.0,
                 min_size: int = 64,
                 min_exposure: int = 30,
                 max_exposure: int = 225):
        """
        Initialize the quality filter.
        
        Args:
            blur_threshold: Minimum Laplacian variance to consider an image not blurry.
            min_size: Minimum width or height in pixels.
            min_exposure: Minimum mean pixel intensity.
            max_exposure: Maximum mean pixel intensity.
        """
        self.blur_threshold = blur_threshold
        self.min_size = min_size
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        
    def check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if an image is blurry using Laplacian variance.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (is_sharp, laplacian_variance).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Check if sharp enough
        is_sharp = variance >= self.blur_threshold
        
        return is_sharp, variance
        
    def check_size(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """
        Check if an image is large enough.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (is_large_enough, (width, height)).
        """
        height, width = image.shape[:2]
        
        # Check if large enough
        is_large_enough = width >= self.min_size and height >= self.min_size
        
        return is_large_enough, (width, height)
        
    def check_exposure(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if an image has good exposure.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (has_good_exposure, mean_intensity).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Compute mean intensity
        mean_intensity = gray.mean()
        
        # Check if exposure is good
        has_good_exposure = self.min_exposure <= mean_intensity <= self.max_exposure
        
        return has_good_exposure, mean_intensity
        
    def check_quality(self, image: np.ndarray) -> Dict:
        """
        Check the quality of an image.
        
        Args:
            image: Input image.
            
        Returns:
            Dictionary with quality check results.
        """
        # Check blur
        is_sharp, variance = self.check_blur(image)
        
        # Check size
        is_large_enough, (width, height) = self.check_size(image)
        
        # Check exposure
        has_good_exposure, mean_intensity = self.check_exposure(image)
        
        # Overall quality
        is_good_quality = is_sharp and is_large_enough and has_good_exposure
        
        # Determine reason for rejection
        reason = None
        if not is_sharp:
            reason = f"blurry (variance: {variance:.2f} < {self.blur_threshold})"
        elif not is_large_enough:
            reason = f"too small (dimensions: {width}x{height}, min: {self.min_size}px)"
        elif not has_good_exposure:
            reason = f"bad exposure (intensity: {mean_intensity:.2f}, range: [{self.min_exposure}, {self.max_exposure}])"
            
        return {
            'is_good_quality': is_good_quality,
            'is_sharp': is_sharp,
            'variance': float(variance),
            'is_large_enough': is_large_enough,
            'dimensions': (width, height),
            'has_good_exposure': has_good_exposure,
            'mean_intensity': float(mean_intensity),
            'reason': reason
        }


def filter_detection(image: np.ndarray,
                    detection: Dict,
                    blur_threshold: float = 100.0,
                    min_size: int = 64,
                    min_exposure: int = 30,
                    max_exposure: int = 225) -> Dict:
    """
    Filter a detection based on the quality of its crop.
    
    Args:
        image: Full image containing the object.
        detection: Detection dictionary with 'bbox' key.
        blur_threshold: Minimum Laplacian variance to consider an image not blurry.
        min_size: Minimum width or height in pixels.
        min_exposure: Minimum mean pixel intensity.
        max_exposure: Maximum mean pixel intensity.
        
    Returns:
        Dictionary with the original detection and added quality check results.
    """
    # Extract crop from image
    # First convert to float, then to int for array indexing
    x1, y1, x2, y2 = map(lambda x: int(float(x)), detection['bbox'])
    crop = image[y1:y2, x1:x2]
    
    # Initialize quality filter
    filter = QualityFilter(blur_threshold, min_size, min_exposure, max_exposure)
    
    # Check quality
    quality_result = filter.check_quality(crop)
    
    # Combine detection and quality result
    result = detection.copy()
    result.update({
        'quality': quality_result,
        'is_good_quality': quality_result['is_good_quality']
    })
    
    return result


def quality_filter(image_path: Union[str, Path],
                  detections: List[Dict],
                  blur_threshold: float = 100.0,
                  min_size: int = 64,
                  min_exposure: int = 30,
                  max_exposure: int = 225) -> List[Dict]:
    """
    Convenience function to filter detections based on quality.
    
    Args:
        image_path: Path to the image file.
        detections: List of detection dictionaries with 'bbox' key.
        blur_threshold: Minimum Laplacian variance to consider an image not blurry.
        min_size: Minimum width or height in pixels.
        min_exposure: Minimum mean pixel intensity.
        max_exposure: Maximum mean pixel intensity.
        
    Returns:
        List of dictionaries with the original detections and added quality check results.
    """
    image_path = Path(image_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    # Filter each detection
    results = []
    for det in detections:
        result = filter_detection(
            image, 
            det, 
            blur_threshold, 
            min_size, 
            min_exposure, 
            max_exposure
        )
        results.append(result)
        
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter detections based on quality")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--detections", type=str, required=True, help="Path to JSON file with detections")
    parser.add_argument("--blur", type=float, default=100.0, help="Blur threshold")
    parser.add_argument("--min-size", type=int, default=64, help="Minimum size")
    parser.add_argument("--min-exposure", type=int, default=30, help="Minimum exposure")
    parser.add_argument("--max-exposure", type=int, default=225, help="Maximum exposure")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        # Load detections
        with open(args.detections, 'r') as f:
            detections = json.load(f)
            
        # Filter detections
        results = quality_filter(
            args.image_path, 
            detections, 
            args.blur, 
            args.min_size, 
            args.min_exposure, 
            args.max_exposure
        )
        
        # Print results
        good_quality = [r for r in results if r['is_good_quality']]
        print(f"Filtered {len(results)} detections, {len(good_quality)} passed quality check:")
        for i, res in enumerate(results):
            quality = res['quality']
            print(f"{i+1}. {res['class']} - Quality: {'PASS' if res['is_good_quality'] else 'FAIL'}")
            if not res['is_good_quality']:
                print(f"   Reason: {quality['reason']}")
                
        # Save to JSON if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
