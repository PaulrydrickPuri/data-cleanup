#!/usr/bin/env python3
"""
Visualization tool for model predictions and dataset inspection.
"""
import os
import cv2
import numpy as np
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetVisualizer:
    """Tool for visualizing datasets and model predictions."""
    
    def __init__(self, 
                 dataset_path: Path, 
                 predictions_path: Optional[Path] = None,
                 class_file: Optional[Path] = None):
        """
        Initialize the visualizer.
        
        Args:
            dataset_path: Path to dataset with images and labels
            predictions_path: Optional path to model prediction results
            class_file: Optional path to class names file
        """
        self.dataset_path = Path(dataset_path)
        self.predictions_path = predictions_path
        self.class_file = class_file
        
        # Find all images in dataset
        self.image_paths = []
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split / 'images'
            if split_dir.exists():
                self.image_paths.extend(list(split_dir.glob('**/*.jpg')))
        
        # Shuffle images for random viewing
        random.shuffle(self.image_paths)
        self.current_index = 0
        
        # Load class names if available
        self.class_names = self._load_class_names()
        
        # Stats
        self.stats = {
            'total_images': len(self.image_paths),
            'viewed_images': 0,
            'total_objects': 0
        }
        
        logger.info(f"Found {len(self.image_paths)} images in dataset")
        
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from file or dataset.yaml."""
        class_names = {}
        
        # Try to load from class file
        if self.class_file and self.class_file.exists():
            try:
                with open(self.class_file, 'r') as f:
                    classes = f.read().splitlines()
                    class_names = {i: name for i, name in enumerate(classes)}
                logger.info(f"Loaded {len(class_names)} class names from {self.class_file}")
            except Exception as e:
                logger.warning(f"Failed to load class names from {self.class_file}: {e}")
        
        # If no classes loaded, try dataset.yaml
        if not class_names:
            yaml_path = self.dataset_path.parent / 'dataset.yaml'
            if yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            class_names = data['names']
                            logger.info(f"Loaded {len(class_names)} class names from {yaml_path}")
                except Exception as e:
                    logger.warning(f"Failed to load class names from {yaml_path}: {e}")
        
        # If still no classes, use generic names
        if not class_names:
            # Look at one label file to determine classes
            label_files = list((self.dataset_path / 'train' / 'labels').glob('*.txt'))
            if label_files:
                try:
                    with open(label_files[0], 'r') as f:
                        classes = set()
                        for line in f.readlines():
                            parts = line.strip().split()
                            if parts:
                                classes.add(int(parts[0]))
                    class_names = {i: f"Class {i}" for i in sorted(classes)}
                    logger.info(f"Generated generic names for {len(class_names)} classes")
                except Exception as e:
                    logger.warning(f"Failed to generate class names: {e}")
        
        return class_names
    
    def _get_label_path(self, image_path: Path) -> Path:
        """Get path to label file for an image."""
        # Replace 'images' with 'labels' and change extension
        parts = list(image_path.parts)
        try:
            idx = parts.index('images')
            parts[idx] = 'labels'
            label_path = Path(*parts).with_suffix('.txt')
            return label_path
        except ValueError:
            # If 'images' not in path, try alternatives
            label_path = image_path.with_suffix('.txt')
            return label_path
    
    def _get_prediction_path(self, image_path: Path) -> Optional[Path]:
        """Get path to prediction file for an image."""
        if self.predictions_path is None:
            return None
            
        # Extract image name and find matching prediction
        img_name = image_path.stem
        pred_files = list(self.predictions_path.glob(f"{img_name}.*"))
        return pred_files[0] if pred_files else None
    
    def _load_labels(self, label_path: Path) -> List[Dict]:
        """Load YOLO format labels."""
        objects = []
        if not label_path.exists():
            return objects
            
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        objects.append({
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, f"Class {class_id}"),
                            'bbox': [x_center, y_center, width, height],
                            'confidence': 1.0  # Ground truth has 100% confidence
                        })
            
            self.stats['total_objects'] += len(objects)
            return objects
        except Exception as e:
            logger.error(f"Error loading labels from {label_path}: {e}")
            return []
    
    def _load_predictions(self, pred_path: Optional[Path]) -> List[Dict]:
        """Load model predictions."""
        if pred_path is None or not pred_path.exists():
            return []
            
        # Determine file format (could be TXT or JSON)
        if pred_path.suffix.lower() == '.txt':
            try:
                objects = []
                with open(pred_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:  # class, x, y, w, h, conf
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            confidence = float(parts[5])
                            
                            objects.append({
                                'class_id': class_id,
                                'class_name': self.class_names.get(class_id, f"Class {class_id}"),
                                'bbox': [x_center, y_center, width, height],
                                'confidence': confidence
                            })
                return objects
            except Exception as e:
                logger.error(f"Error loading predictions from {pred_path}: {e}")
                return []
        else:
            # Could implement JSON format later if needed
            logger.warning(f"Unsupported prediction format: {pred_path.suffix}")
            return []
    
    def _draw_bbox(self, img: np.ndarray, 
                  bbox: List[float], 
                  class_name: str, 
                  confidence: float,
                  color: Tuple[int, int, int],
                  is_prediction: bool = False) -> np.ndarray:
        """Draw bounding box on image."""
        h, w = img.shape[:2]
        x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to absolute
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name} {confidence:.2f}" if is_prediction else class_name
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
    
    def visualize_image(self, img_path: Path) -> np.ndarray:
        """Visualize an image with its labels and predictions."""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get label and prediction paths
        label_path = self._get_label_path(img_path)
        pred_path = self._get_prediction_path(img_path)
        
        # Load ground truth labels
        gt_objects = self._load_labels(label_path)
        
        # Load predictions if available
        pred_objects = self._load_predictions(pred_path)
        
        # Draw ground truth boxes (in green)
        for obj in gt_objects:
            img = self._draw_bbox(
                img, 
                obj['bbox'], 
                obj['class_name'], 
                obj['confidence'], 
                (0, 255, 0)
            )
        
        # Draw prediction boxes (in blue)
        for obj in pred_objects:
            img = self._draw_bbox(
                img, 
                obj['bbox'], 
                obj['class_name'], 
                obj['confidence'], 
                (255, 0, 0),
                is_prediction=True
            )
        
        # Add image info
        img_info = f"Image: {img_path.name} | GT: {len(gt_objects)} objects"
        if pred_objects:
            img_info += f" | Pred: {len(pred_objects)} objects"
        
        # Add text overlay with image info
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, img_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.stats['viewed_images'] += 1
        return img
    
    def run_interactive_viewer(self) -> None:
        """Run interactive matplotlib viewer."""
        try:
            if not self.image_paths:
                logger.error("No images found in dataset")
                return
                
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.subplots_adjust(bottom=0.2)
            
            # First image
            img_path = self.image_paths[self.current_index]
            img = self.visualize_image(img_path)
            img_display = ax.imshow(img)
            
            # Add navigation buttons
            ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
            ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
            btn_prev = Button(ax_prev, 'Previous')
            btn_next = Button(ax_next, 'Next')
            
            # Add slider for random jumping
            ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
            slider = Slider(ax_slider, 'Image', 0, len(self.image_paths)-1, valinit=0, valstep=1)
            
            def update_image(index):
                if 0 <= index < len(self.image_paths):
                    self.current_index = index
                    img_path = self.image_paths[self.current_index]
                    img = self.visualize_image(img_path)
                    img_display.set_data(img)
                    slider.set_val(self.current_index)
                    plt.draw()
            
            def on_prev(event):
                update_image(self.current_index - 1)
            
            def on_next(event):
                update_image(self.current_index + 1)
            
            def on_slide(val):
                update_image(int(val))
            
            btn_prev.on_clicked(on_prev)
            btn_next.on_clicked(on_next)
            slider.on_changed(on_slide)
            
            plt.show()
            
            # Show stats when done
            logger.info(f"Viewed {self.stats['viewed_images']} of {self.stats['total_images']} images")
            logger.info(f"Dataset contains {self.stats['total_objects']} objects")
            
        except Exception as e:
            logger.error(f"Error in interactive viewer: {e}")
    
    def export_visualizations(self, output_dir: Path, num_images: int = 10) -> None:
        """Export visualizations of dataset images to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select a subset of images
        subset = self.image_paths[:num_images] if num_images > 0 else self.image_paths
        
        for i, img_path in enumerate(subset):
            try:
                img = self.visualize_image(img_path)
                output_path = output_dir / f"vis_{img_path.stem}.jpg"
                
                # Convert RGB back to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_bgr)
                logger.info(f"Saved visualization to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting visualization for {img_path}: {e}")
        
        logger.info(f"Exported {len(subset)} visualizations to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset and model predictions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--predictions", type=str, default=None, help="Path to model predictions directory")
    parser.add_argument("--classes", type=str, default=None, help="Path to class names file")
    parser.add_argument("--export", type=str, default=None, help="Export visualizations to directory")
    parser.add_argument("--num-export", type=int, default=10, help="Number of images to export (if --export is used)")
    
    args = parser.parse_args()
    
    visualizer = DatasetVisualizer(
        dataset_path=Path(args.dataset),
        predictions_path=Path(args.predictions) if args.predictions else None,
        class_file=Path(args.classes) if args.classes else None
    )
    
    if args.export:
        visualizer.export_visualizations(Path(args.export), args.num_export)
    else:
        visualizer.run_interactive_viewer()

if __name__ == "__main__":
    main()
