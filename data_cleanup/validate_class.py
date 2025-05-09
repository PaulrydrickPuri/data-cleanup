#!/usr/bin/env python3
"""
Validate detected object classes using CLIP embeddings and cosine similarity.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
import torch
import clip
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassValidator:
    """Validate object classes using CLIP embeddings and cosine similarity."""
    
    def __init__(self, 
                 anchors_path: Union[str, Path],
                 similarity_threshold: float = 0.65,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the class validator.
        
        Args:
            anchors_path: Path to JSON file with ground truth anchor embeddings.
            similarity_threshold: Minimum cosine similarity to consider a match valid.
            device: Device to run CLIP on ('cuda' or 'cpu').
        """
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Load CLIP model
        logger.info(f"Loading CLIP ViT-B/32 model on {device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load anchor embeddings
        self.anchors = self._load_anchors(anchors_path)
        
    def _load_anchors(self, anchors_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load anchor embeddings from JSON file.
        
        Args:
            anchors_path: Path to JSON file with ground truth anchor embeddings.
            
        Returns:
            Dictionary mapping class names to anchor embeddings.
        """
        anchors_path = Path(anchors_path)
        
        if not anchors_path.exists():
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")
            
        with open(anchors_path, 'r') as f:
            anchors_data = json.load(f)
            
        anchors = {}
        
        for class_name, embeddings in anchors_data.items():
            # Convert list to tensor
            if isinstance(embeddings, list):
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
                # Normalize if not already normalized
                if torch.norm(embeddings_tensor).item() != 1.0:
                    embeddings_tensor = embeddings_tensor / torch.norm(embeddings_tensor)
                    
                anchors[class_name] = embeddings_tensor.to(self.device)
            else:
                logger.warning(f"Invalid embedding format for class {class_name}")
                
        logger.info(f"Loaded anchor embeddings for {len(anchors)} classes")
        return anchors
        
    def compute_embedding(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Compute CLIP embedding for an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV) or PIL Image.
            
        Returns:
            Normalized embedding tensor.
        """
        # Convert OpenCV image to PIL if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if image.shape[2] == 3:  # Check if it has 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
        # Preprocess for CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Compute embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.squeeze()
        
    def validate(self, 
                image: Union[np.ndarray, Image.Image], 
                bbox: List[int], 
                class_name: str) -> Dict:
        """
        Validate if the detected object matches the claimed class.
        
        Args:
            image: Full image containing the object.
            bbox: Bounding box of the object in [x1, y1, x2, y2] format.
            class_name: Claimed class name of the object.
            
        Returns:
            Dictionary with validation results including:
                - 'valid': Whether the object is valid (bool)
                - 'top_class': The class with highest similarity
                - 'similarity': Cosine similarity score
                - 'class_match': Whether top_class matches claimed class
        """
        if class_name not in self.anchors:
            logger.warning(f"No anchor embedding for class '{class_name}'")
            return {
                'valid': False,
                'top_class': None,
                'similarity': 0.0,
                'class_match': False
            }
            
        # Extract crop from image
        if isinstance(image, np.ndarray):
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
        else:
            # PIL Image
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))
            
        # Compute embedding for crop
        crop_embedding = self.compute_embedding(crop)
        
        # Compute similarities with all anchors
        similarities = {}
        for cls, anchor_emb in self.anchors.items():
            similarity = torch.dot(crop_embedding, anchor_emb).item()
            similarities[cls] = similarity
            
        # Find top class
        top_class = max(similarities, key=similarities.get)
        top_similarity = similarities[top_class]
        
        # Check if claimed class matches top class
        class_match = (top_class == class_name)
        
        # Check if similarity is above threshold
        valid = class_match and (top_similarity >= self.similarity_threshold)
        
        return {
            'valid': valid,
            'top_class': top_class,
            'similarity': top_similarity,
            'similarities': similarities,
            'class_match': class_match
        }


def validate_class(image_path: Union[str, Path],
                  detection: Dict,
                  anchors_path: Union[str, Path],
                  similarity_threshold: float = 0.65) -> Dict:
    """
    Convenience function to validate a detected object's class.
    
    Args:
        image_path: Path to the image file.
        detection: Detection dictionary with 'bbox' and 'class' keys.
        anchors_path: Path to JSON file with ground truth anchor embeddings.
        similarity_threshold: Minimum cosine similarity to consider a match valid.
        
    Returns:
        Dictionary with the original detection and added validation results.
    """
    image_path = Path(image_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    # Initialize validator
    validator = ClassValidator(anchors_path, similarity_threshold)
    
    # Validate class
    validation_result = validator.validate(
        image, 
        detection['bbox'], 
        detection['class']
    )
    
    # Combine detection and validation result
    result = detection.copy()
    result.update({
        'validation': validation_result
    })
    
    return result


def generate_anchors(image_paths: List[Union[str, Path]],
                    class_labels: List[str],
                    output_path: Union[str, Path],
                    device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    """
    Generate anchor embeddings from ground truth images.
    
    Args:
        image_paths: List of paths to ground truth images.
        class_labels: Corresponding class labels for each image.
        output_path: Path to save the anchor embeddings JSON file.
        device: Device to run CLIP on ('cuda' or 'cpu').
    """
    assert len(image_paths) == len(class_labels), "Number of images and labels must match"
    
    # Load CLIP model
    logger.info(f"Loading CLIP ViT-B/32 model on {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Compute embeddings for each image
    embeddings = {}
    
    for i, (image_path, class_label) in enumerate(zip(image_paths, class_labels)):
        image_path = Path(image_path)
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            continue
            
        # Preprocess for CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Compute embedding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Add to embeddings
        if class_label not in embeddings:
            embeddings[class_label] = []
            
        embeddings[class_label].append(image_features.squeeze().cpu().tolist())
        
    # Average embeddings for each class
    averaged_embeddings = {}
    
    for class_label, class_embeddings in embeddings.items():
        if not class_embeddings:
            continue
            
        # Convert to tensor
        class_embeddings_tensor = torch.tensor(class_embeddings)
        
        # Average
        avg_embedding = torch.mean(class_embeddings_tensor, dim=0)
        
        # Normalize
        avg_embedding = avg_embedding / torch.norm(avg_embedding)
        
        # Add to averaged embeddings
        averaged_embeddings[class_label] = avg_embedding.tolist()
        
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(averaged_embeddings, f, indent=2)
        
    logger.info(f"Saved anchor embeddings for {len(averaged_embeddings)} classes to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate object classes using CLIP embeddings")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate anchors command
    gen_parser = subparsers.add_parser("generate", help="Generate anchor embeddings")
    gen_parser.add_argument("--images", type=str, required=True, help="Path to text file with image paths (one per line)")
    gen_parser.add_argument("--labels", type=str, required=True, help="Path to text file with class labels (one per line)")
    gen_parser.add_argument("--output", type=str, default="assets/gt_anchors.json", help="Output JSON file path")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate object classes")
    val_parser.add_argument("image_path", type=str, help="Path to the image file")
    val_parser.add_argument("--detections", type=str, required=True, help="Path to JSON file with detections")
    val_parser.add_argument("--anchors", type=str, default="assets/gt_anchors.json", help="Path to anchor embeddings JSON file")
    val_parser.add_argument("--threshold", type=float, default=0.65, help="Similarity threshold")
    val_parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        if args.command == "generate":
            # Load image paths and labels
            with open(args.images, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
                
            with open(args.labels, 'r') as f:
                class_labels = [line.strip() for line in f.readlines()]
                
            # Generate anchors
            generate_anchors(image_paths, class_labels, args.output)
            
        elif args.command == "validate":
            # Load detections
            with open(args.detections, 'r') as f:
                detections = json.load(f)
                
            # Validate each detection
            results = []
            for det in detections:
                result = validate_class(
                    args.image_path, 
                    det, 
                    args.anchors, 
                    args.threshold
                )
                results.append(result)
                
            # Print results
            print(f"Validated {len(results)} detections:")
            for i, res in enumerate(results):
                valid = res['validation']['valid']
                top_class = res['validation']['top_class']
                similarity = res['validation']['similarity']
                print(f"{i+1}. {res['class']} -> {top_class} (similarity: {similarity:.2f}, valid: {valid})")
                
            # Save to JSON if output path provided
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
