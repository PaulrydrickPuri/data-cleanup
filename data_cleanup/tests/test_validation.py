#!/usr/bin/env python3
"""
Tests for the class validation module.
"""
import os
import sys
import json
import pytest
from pathlib import Path
import cv2
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_cleanup.validate_class import validate_class, ClassValidator, generate_anchors


class TestValidation:
    """Test cases for class validation."""
    
    @pytest.fixture
    def sample_anchors(self, tmp_path):
        """Create sample anchor embeddings for testing."""
        anchors_path = tmp_path / "test_anchors.json"
        
        # Create sample embeddings (normalized)
        anchors = {
            "vehicle": [0.1] * 512,
            "carplate": [0.2] * 512,
            "logo": [0.3] * 512
        }
        
        # Normalize embeddings
        for cls, emb in anchors.items():
            norm = np.sqrt(sum([x*x for x in emb]))
            anchors[cls] = [x/norm for x in emb]
            
        with open(anchors_path, 'w') as f:
            json.dump(anchors, f)
            
        return anchors_path
        
    @pytest.fixture
    def sample_detection(self):
        """Create a sample detection for testing."""
        return {
            "class": "vehicle",
            "bbox": [10, 20, 110, 120],
            "score": 0.9
        }
        
    def test_validator_initialization(self, sample_anchors):
        """Test that the validator initializes correctly."""
        validator = ClassValidator(sample_anchors)
        assert validator is not None
        assert validator.similarity_threshold == 0.65
        assert len(validator.anchors) == 3
        assert "vehicle" in validator.anchors
        assert "carplate" in validator.anchors
        assert "logo" in validator.anchors
        
    def test_validator_custom_threshold(self, sample_anchors):
        """Test that the validator respects custom similarity threshold."""
        threshold = 0.8
        validator = ClassValidator(sample_anchors, similarity_threshold=threshold)
        assert validator.similarity_threshold == threshold
        
    def test_compute_embedding(self, sample_anchors, monkeypatch):
        """Test computing embeddings for an image."""
        # Mock CLIP model
        class MockCLIP:
            def encode_image(self, image):
                # Return a fixed embedding
                return torch.ones((1, 512), dtype=torch.float32)
                
        class MockModel:
            def __init__(self):
                self.model = MockCLIP()
                
            def load(self, model_name, device):
                # Return mock model and preprocess function
                def preprocess(image):
                    return torch.zeros((1, 3, 224, 224))
                return self.model, preprocess
                
        # Apply monkeypatch
        monkeypatch.setattr("data_cleanup.validate_class.clip", MockModel())
        
        # Create validator
        validator = ClassValidator(sample_anchors)
        
        # Create test image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Compute embedding
        embedding = validator.compute_embedding(image)
        
        # Check embedding
        assert embedding.shape == (512,)
        assert torch.allclose(embedding, torch.ones(512) / torch.sqrt(torch.tensor(512.0)))
        
    def test_validate_class_function(self, sample_anchors, sample_detection, monkeypatch):
        """Test the validate_class convenience function."""
        # Mock cv2.imread to return a valid image
        def mock_imread(path):
            return np.zeros((300, 300, 3), dtype=np.uint8)
            
        # Mock ClassValidator.validate to return a fixed result
        def mock_validate(self, image, bbox, class_name):
            return {
                'valid': True,
                'top_class': class_name,
                'similarity': 0.9,
                'similarities': {class_name: 0.9},
                'class_match': True
            }
            
        # Apply monkeypatches
        monkeypatch.setattr(cv2, "imread", mock_imread)
        monkeypatch.setattr(ClassValidator, "validate", mock_validate)
        
        # Test validation
        result = validate_class("test_image.jpg", sample_detection, sample_anchors)
        
        # Check result
        assert "validation" in result
        assert result["validation"]["valid"] is True
        assert result["validation"]["top_class"] == "vehicle"
        assert result["validation"]["similarity"] == 0.9
        assert result["validation"]["class_match"] is True
        
    def test_invalid_class(self, sample_anchors, monkeypatch):
        """Test validation with an invalid class."""
        # Create validator
        validator = ClassValidator(sample_anchors)
        
        # Create test image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Mock compute_embedding to return a fixed embedding
        def mock_compute_embedding(self, image):
            # Return an embedding that's closer to "logo" than "vehicle"
            emb = torch.zeros(512, dtype=torch.float32)
            emb[0] = 1.0  # This will be normalized
            return emb
            
        # Apply monkeypatch
        monkeypatch.setattr(ClassValidator, "compute_embedding", mock_compute_embedding)
        
        # Test validation with a mismatched class
        result = validator.validate(image, [10, 20, 110, 120], "vehicle")
        
        # The top class should not be "vehicle"
        assert result["valid"] is False
        assert result["top_class"] != "vehicle"
        assert result["class_match"] is False
        
    @pytest.mark.parametrize("class_name,expected_valid", [
        ("vehicle", True),
        ("carplate", False),
        ("logo", False)
    ])
    def test_class_validation_swapped_labels(self, sample_anchors, monkeypatch, class_name, expected_valid):
        """
        Test that intentionally swapped labels are rejected.
        This test verifies that validation correctly identifies mismatched classes.
        """
        # Create validator
        validator = ClassValidator(sample_anchors)
        
        # Create test image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Mock compute_embedding to return a fixed embedding for "vehicle"
        def mock_compute_embedding(self, image):
            # Return an embedding that's closest to "vehicle"
            emb = torch.zeros(512, dtype=torch.float32)
            # Make it very similar to the vehicle anchor
            emb[0] = 0.1
            emb = emb / torch.norm(emb)
            return emb
            
        # Apply monkeypatch
        monkeypatch.setattr(ClassValidator, "compute_embedding", mock_compute_embedding)
        
        # Test validation with different claimed classes
        result = validator.validate(image, [10, 20, 110, 120], class_name)
        
        # Check if the validation result matches expectations
        assert result["valid"] is expected_valid
        assert result["top_class"] == "vehicle"  # Should always be vehicle
        assert result["class_match"] == expected_valid
        
    def test_generate_anchors(self, tmp_path, monkeypatch):
        """Test generating anchor embeddings from ground truth images."""
        # Create test directory with sample images
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        
        # Create image paths and labels
        image_paths = [test_dir / f"test_image_{i}.jpg" for i in range(3)]
        class_labels = ["vehicle", "carplate", "logo"]
        
        # Create test images
        for path in image_paths:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(path), img)
            
        # Mock CLIP model
        class MockCLIP:
            def encode_image(self, image):
                # Return different embeddings based on the batch index
                batch_size = image.shape[0]
                return torch.ones((batch_size, 512), dtype=torch.float32)
                
        class MockModel:
            def __init__(self):
                self.model = MockCLIP()
                
            def load(self, model_name, device):
                # Return mock model and preprocess function
                def preprocess(image):
                    return torch.zeros((1, 3, 224, 224))
                return self.model, preprocess
                
        # Apply monkeypatch
        monkeypatch.setattr("data_cleanup.validate_class.clip", MockModel())
        
        # Generate anchors
        output_path = tmp_path / "generated_anchors.json"
        generate_anchors(image_paths, class_labels, output_path)
        
        # Check that the anchors file was created
        assert output_path.exists()
        
        # Load and check anchors
        with open(output_path, 'r') as f:
            anchors = json.load(f)
            
        assert len(anchors) == 3
        assert "vehicle" in anchors
        assert "carplate" in anchors
        assert "logo" in anchors
        assert len(anchors["vehicle"]) == 512


if __name__ == "__main__":
    pytest.main(["-v", __file__])
