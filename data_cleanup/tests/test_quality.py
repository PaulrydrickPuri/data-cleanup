#!/usr/bin/env python3
"""
Tests for the quality filter module.
"""
import os
import sys
import pytest
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_cleanup.quality_filter import QualityFilter, filter_detection, quality_filter


class TestQualityFilter:
    """Test cases for quality filtering."""
    
    @pytest.fixture
    def sample_detection(self):
        """Create a sample detection for testing."""
        return {
            "class": "vehicle",
            "bbox": [10, 20, 110, 120],
            "score": 0.9
        }
        
    def test_filter_initialization(self):
        """Test that the filter initializes correctly."""
        filter = QualityFilter()
        assert filter is not None
        assert filter.blur_threshold == 100.0
        assert filter.min_size == 64
        assert filter.min_exposure == 30
        assert filter.max_exposure == 225
        
    def test_filter_custom_parameters(self):
        """Test that the filter respects custom parameters."""
        filter = QualityFilter(
            blur_threshold=200.0,
            min_size=128,
            min_exposure=50,
            max_exposure=200
        )
        assert filter.blur_threshold == 200.0
        assert filter.min_size == 128
        assert filter.min_exposure == 50
        assert filter.max_exposure == 200
        
    def test_check_blur(self):
        """Test blur detection."""
        filter = QualityFilter(blur_threshold=100.0)
        
        # Create a sharp image (high variance)
        sharp_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        is_sharp, variance = filter.check_blur(sharp_img)
        
        # Create a blurry image (low variance)
        blurry_img = np.ones((100, 100), dtype=np.uint8) * 128
        is_blurry, blurry_variance = filter.check_blur(blurry_img)
        
        # Check results
        assert variance > blurry_variance
        assert is_sharp or not is_blurry  # At least one should be true
        
    def test_check_size(self):
        """Test size checking."""
        filter = QualityFilter(min_size=64)
        
        # Create images of different sizes
        small_img = np.zeros((32, 32, 3), dtype=np.uint8)
        large_img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Check sizes
        is_small_enough, small_dims = filter.check_size(small_img)
        is_large_enough, large_dims = filter.check_size(large_img)
        
        # Check results
        assert not is_small_enough
        assert is_large_enough
        assert small_dims == (32, 32)
        assert large_dims == (128, 128)
        
    def test_check_exposure(self):
        """Test exposure checking."""
        filter = QualityFilter(min_exposure=30, max_exposure=225)
        
        # Create images with different exposures
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 10
        good_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        
        # Check exposures
        is_dark_good, dark_intensity = filter.check_exposure(dark_img)
        is_good_good, good_intensity = filter.check_exposure(good_img)
        is_bright_good, bright_intensity = filter.check_exposure(bright_img)
        
        # Check results
        assert not is_dark_good
        assert is_good_good
        assert not is_bright_good
        assert dark_intensity < filter.min_exposure
        assert filter.min_exposure <= good_intensity <= filter.max_exposure
        assert bright_intensity > filter.max_exposure
        
    def test_check_quality(self):
        """Test overall quality checking."""
        filter = QualityFilter()
        
        # Create a good quality image
        good_img = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        
        # Create a poor quality image (too small)
        small_img = np.random.randint(100, 200, (32, 32, 3), dtype=np.uint8)
        
        # Check quality
        good_result = filter.check_quality(good_img)
        small_result = filter.check_quality(small_img)
        
        # Check results
        assert "is_good_quality" in good_result
        assert "is_sharp" in good_result
        assert "is_large_enough" in good_result
        assert "has_good_exposure" in good_result
        
        assert small_result["is_large_enough"] is False
        assert small_result["is_good_quality"] is False
        assert "reason" in small_result and small_result["reason"] is not None
        
    def test_filter_detection_function(self, sample_detection):
        """Test the filter_detection function."""
        # Create a test image
        image = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
        
        # Filter detection
        result = filter_detection(image, sample_detection)
        
        # Check result
        assert "quality" in result
        assert "is_good_quality" in result
        assert "is_sharp" in result["quality"]
        assert "is_large_enough" in result["quality"]
        assert "has_good_exposure" in result["quality"]
        
    def test_quality_filter_function(self, sample_detection, monkeypatch):
        """Test the quality_filter convenience function."""
        # Mock cv2.imread to return a valid image
        def mock_imread(path):
            return np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
            
        # Apply monkeypatch
        monkeypatch.setattr(cv2, "imread", mock_imread)
        
        # Test quality filter
        results = quality_filter("test_image.jpg", [sample_detection])
        
        # Check results
        assert len(results) == 1
        assert "quality" in results[0]
        assert "is_good_quality" in results[0]
        
    @pytest.mark.parametrize("blur_level,expected_result", [
        (0, False),    # Very blurry
        (150, True)    # Sharp
    ])
    def test_blur_detection(self, blur_level, expected_result):
        """
        Test that blurred images are correctly identified and discarded.
        This test verifies that the quality filter correctly identifies blurry images.
        """
        # Create a filter with a threshold of 100
        filter = QualityFilter(blur_threshold=100.0)
        
        # Create an image with controlled blur
        if blur_level == 0:
            # Create a completely uniform image (zero variance)
            image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        else:
            # Create a noisy image (high variance)
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
        # Check quality
        result = filter.check_quality(image)
        
        # Check if the blur detection matches expectations
        assert result["is_sharp"] == expected_result
        
        # If the image is blurry, the overall quality should be poor
        if not expected_result:
            assert result["is_good_quality"] is False
            assert "blurry" in result["reason"].lower()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
