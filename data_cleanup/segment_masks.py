#!/usr/bin/env python3
"""
Segment masks for detected objects using Segment Anything Model (SAM).
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MaskSegmenter:
    """Segment Anything Model (SAM) for object segmentation."""
    
    def __init__(self, sam_checkpoint: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the SAM segmenter.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint. If None, downloads the default model.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        
        # Default to ViT-B SAM model if no checkpoint provided
        if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
            # Look for the checkpoint in the torch hub directory
            home_dir = os.path.expanduser("~")
            torch_hub_dir = os.path.join(home_dir, ".cache", "torch", "hub", "checkpoints")
            sam_checkpoint = os.path.join(torch_hub_dir, "sam_vit_b_01ec64.pth")
            
            if not os.path.exists(sam_checkpoint):
                logger.info("Downloading SAM ViT-B checkpoint...")
                # This will trigger the automatic download
                
        # Determine model type from checkpoint name
        if "vit_h" in sam_checkpoint:
            model_type = "vit_h"
        elif "vit_l" in sam_checkpoint:
            model_type = "vit_l"
        else:
            model_type = "vit_b"  # Default
            
        logger.info(f"Loading SAM {model_type} model from {sam_checkpoint} on {device}")
        
        # Initialize SAM
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
    def segment(self, image: np.ndarray, bboxes: List[List[int]]) -> List[Dict]:
        """
        Generate masks for objects defined by bounding boxes.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            bboxes: List of bounding boxes in [x1, y1, x2, y2] format.
            
        Returns:
            List of dictionaries with keys 'mask', 'bbox', 'area', and 'score'.
            'mask' is a binary numpy array of the same size as the input image.
        """
        # Convert BGR to RGB for SAM
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(rgb_image)
        
        masks = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            # Convert to SAM input format (center point, width, height)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Get SAM input prompt
            input_box = np.array([x1, y1, x2, y2])
            
            # Generate mask
            masks_result, scores, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False  # Only return the best mask
            )
            
            # Get the best mask (first one since multimask_output=False)
            mask = masks_result[0]
            score = float(scores[0])
            
            # Calculate area
            area = int(mask.sum())
            
            masks.append({
                'mask': mask,
                'bbox': bbox,
                'area': area,
                'score': score
            })
            
        return masks
        
    def save_masks(self, masks: List[Dict], output_dir: Union[str, Path], 
                  image_name: str, compress: bool = True) -> List[str]:
        """
        Save masks as PNG files.
        
        Args:
            masks: List of mask dictionaries from segment().
            output_dir: Directory to save masks to.
            image_name: Base name for the mask files.
            compress: Whether to compress the masks.
            
        Returns:
            List of paths to saved mask files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['mask']
            
            # Convert boolean mask to uint8 (0 or 255)
            mask_img = (mask * 255).astype(np.uint8)
            
            # Save path
            mask_filename = f"{image_name}_mask_{i:03d}.png"
            mask_path = output_dir / mask_filename
            
            # Save mask
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9] if compress else []
            cv2.imwrite(str(mask_path), mask_img, compression_params)
            
            saved_paths.append(str(mask_path))
            
        return saved_paths


def segment_masks(image_path: Union[str, Path], 
                 detections: List[Dict],
                 output_dir: Union[str, Path] = "masks",
                 sam_checkpoint: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to segment masks for detected objects.
    
    Args:
        image_path: Path to the image file.
        detections: List of detection dictionaries with 'bbox' key.
        output_dir: Directory to save masks to.
        sam_checkpoint: Path to SAM model checkpoint.
        
    Returns:
        List of dictionaries with detection info and added 'mask_path' key.
    """
    image_path = Path(image_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    # Extract bboxes from detections
    bboxes = [det['bbox'] for det in detections]
    
    # Initialize segmenter
    segmenter = MaskSegmenter(sam_checkpoint)
    
    # Generate masks
    mask_results = segmenter.segment(image, bboxes)
    
    # Save masks
    image_name = image_path.stem
    mask_paths = segmenter.save_masks(mask_results, output_dir, image_name)
    
    # Add mask paths to detections
    results = []
    for i, (det, mask_dict, mask_path) in enumerate(zip(detections, mask_results, mask_paths)):
        result = det.copy()
        result.update({
            'mask_path': mask_path,
            'mask_score': mask_dict['score'],
            'mask_area': mask_dict['area']
        })
        results.append(result)
        
    return results


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Segment masks for detected objects using SAM")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--detections", type=str, required=True, help="Path to JSON file with detections")
    parser.add_argument("--output-dir", type=str, default="masks", help="Directory to save masks to")
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="Path to SAM model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        # Load detections
        with open(args.detections, 'r') as f:
            detections = json.load(f)
            
        # Segment masks
        results = segment_masks(
            args.image_path, 
            detections, 
            args.output_dir, 
            args.sam_checkpoint
        )
        
        # Print results
        print(f"Generated {len(results)} masks:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['class']} mask saved to {res['mask_path']} (score: {res['mask_score']:.2f})")
            
        # Save to JSON if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
