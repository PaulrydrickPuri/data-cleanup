#!/usr/bin/env python3
"""
OCR check for license plates using EasyOCR and regex validation.
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
import easyocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRChecker:
    """Check license plates using EasyOCR and regex validation."""
    
    def __init__(self, 
                 regex_path: Union[str, Path],
                 gpu: bool = True,
                 languages: List[str] = ['en']):
        """
        Initialize the OCR checker.
        
        Args:
            regex_path: Path to JSON file with regex patterns.
            gpu: Whether to use GPU for OCR.
            languages: Languages to use for OCR.
        """
        self.regex_patterns = self._load_regex_patterns(regex_path)
        
        # Initialize EasyOCR reader
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def _load_regex_patterns(self, regex_path: Union[str, Path]) -> Dict[str, str]:
        """
        Load regex patterns from JSON file.
        
        Args:
            regex_path: Path to JSON file with regex patterns.
            
        Returns:
            Dictionary mapping pattern names to regex patterns.
        """
        regex_path = Path(regex_path)
        
        if not regex_path.exists():
            raise FileNotFoundError(f"Regex patterns file not found: {regex_path}")
            
        with open(regex_path, 'r') as f:
            patterns = json.load(f)
            
        # Compile regex patterns
        compiled_patterns = {}
        for name, pattern in patterns.items():
            try:
                compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
                
        logger.info(f"Loaded {len(compiled_patterns)} regex patterns")
        return compiled_patterns
        
    def preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR results.
        
        Args:
            image: License plate image.
            
        Returns:
            Preprocessed image.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
        
    def check_plate(self, 
                   image: np.ndarray, 
                   bbox: List[int],
                   preprocess: bool = True) -> Dict:
        """
        Check if a license plate is valid using OCR and regex validation.
        
        Args:
            image: Full image containing the license plate.
            bbox: Bounding box of the license plate in [x1, y1, x2, y2] format.
            preprocess: Whether to preprocess the plate image.
            
        Returns:
            Dictionary with OCR results including:
                - 'text': Recognized text
                - 'confidence': OCR confidence
                - 'verified': Whether the text matches any regex pattern
                - 'pattern_name': Name of the matching pattern (if any)
        """
        # Extract plate from image
        x1, y1, x2, y2 = bbox
        plate_img = image[y1:y2, x1:x2]
        
        # Preprocess plate image
        if preprocess:
            plate_img = self.preprocess_plate(plate_img)
            
        # Perform OCR
        results = self.reader.readtext(plate_img)
        
        # No text detected
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'verified': False,
                'pattern_name': None,
                'hash': None  # No PII stored
            }
            
        # Get text with highest confidence
        text = ''
        confidence = 0.0
        
        for result in results:
            if len(result) >= 2:
                bbox, text_result = result[:2]
                if len(result) >= 3:
                    conf = result[2]
                    if conf > confidence:
                        text = text_result
                        confidence = conf
                else:
                    # If no confidence provided, use the first result
                    text = text_result
                    break
                    
        # Remove spaces and normalize
        text = text.strip().upper()
        
        # Check against regex patterns
        verified = False
        pattern_name = None
        
        for name, pattern in self.regex_patterns.items():
            if pattern.match(text):
                verified = True
                pattern_name = name
                break
                
        # Hash the text to avoid storing PII
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest() if text else None
        
        return {
            'text': '',  # Don't store the actual text for privacy
            'confidence': float(confidence),
            'verified': verified,
            'pattern_name': pattern_name,
            'hash': text_hash
        }


def ocr_check(image_path: Union[str, Path],
             detection: Dict,
             regex_path: Union[str, Path],
             preprocess: bool = True) -> Dict:
    """
    Convenience function to check a license plate using OCR.
    
    Args:
        image_path: Path to the image file.
        detection: Detection dictionary with 'bbox' and 'class' keys.
        regex_path: Path to JSON file with regex patterns.
        preprocess: Whether to preprocess the plate image.
        
    Returns:
        Dictionary with the original detection and added OCR results.
    """
    # Only process carplates
    if detection['class'] != 'carplate':
        return detection
        
    image_path = Path(image_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    # Initialize OCR checker
    checker = OCRChecker(regex_path)
    
    # Check plate
    ocr_result = checker.check_plate(
        image, 
        detection['bbox'], 
        preprocess
    )
    
    # Combine detection and OCR result
    result = detection.copy()
    result.update({
        'ocr': ocr_result,
        'ocr_verified': ocr_result['verified']
    })
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check license plates using OCR")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--detections", type=str, required=True, help="Path to JSON file with detections")
    parser.add_argument("--regex", type=str, default="assets/regex_plate.json", help="Path to regex patterns JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        # Load detections
        with open(args.detections, 'r') as f:
            detections = json.load(f)
            
        # Check each carplate
        results = []
        for det in detections:
            result = det.copy()
            
            if det['class'] == 'carplate':
                result = ocr_check(
                    args.image_path, 
                    det, 
                    args.regex
                )
                
            results.append(result)
            
        # Print results
        print(f"Checked {len([r for r in results if r['class'] == 'carplate'])} carplates:")
        for i, res in enumerate(results):
            if res['class'] == 'carplate':
                verified = res['ocr_verified']
                confidence = res['ocr']['confidence']
                print(f"{i+1}. Carplate (confidence: {confidence:.2f}, verified: {verified})")
                
        # Save to JSON if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
