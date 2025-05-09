#!/usr/bin/env python3
"""
Script to generate CLIP embeddings for ground truth images.
These embeddings will serve as anchors for validating the classes of detected objects.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_clip_model(model_name="ViT-B/32"):
    """Load CLIP model and return model and preprocessing function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def generate_embeddings(model, preprocess, device, image_paths, batch_size=16):
    """Generate CLIP embeddings for a list of images."""
    embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                processed_image = preprocess(image)
                batch_images.append(processed_image)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Add a placeholder embedding
                batch_images.append(torch.zeros_like(processed_image))
        
        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
            batch_embeddings = batch_embeddings.cpu().numpy()
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def generate_class_embeddings(model, preprocess, device, class_names):
    """Generate text embeddings for class names."""
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    
    with torch.no_grad():
        text_embeddings = model.encode_text(text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings.cpu().numpy()
    
    return {class_name: embedding for class_name, embedding in zip(class_names, text_embeddings)}

def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for ground truth images")
    parser.add_argument("--images", type=str, required=True, help="Path to text file with image paths")
    parser.add_argument("--labels", type=str, required=True, help="Path to text file with class labels")
    parser.add_argument("--output", type=str, default="gt_anchors.json", help="Output JSON file for embeddings")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing images")
    
    args = parser.parse_args()
    
    # Load image paths and labels
    with open(args.images, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    with open(args.labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    if len(image_paths) != len(labels):
        print(f"Error: Number of images ({len(image_paths)}) does not match number of labels ({len(labels)})")
        sys.exit(1)
    
    # Load CLIP model
    print(f"Loading CLIP model {args.model}...")
    model, preprocess, device = load_clip_model(args.model)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(image_paths)} images...")
    image_embeddings = generate_embeddings(model, preprocess, device, image_paths, args.batch_size)
    
    # Get unique class names
    unique_classes = sorted(set(labels))
    print(f"Found {len(unique_classes)} unique classes: {', '.join(unique_classes)}")
    
    # Generate text embeddings for class names
    print("Generating text embeddings for class names...")
    text_embeddings = generate_class_embeddings(model, preprocess, device, unique_classes)
    
    # Group embeddings by class
    class_embeddings = {cls: [] for cls in unique_classes}
    class_paths = {cls: [] for cls in unique_classes}
    
    for img_path, label, embedding in zip(image_paths, labels, image_embeddings):
        class_embeddings[label].append(embedding.tolist())
        class_paths[label].append(img_path)
    
    # Calculate average embedding for each class
    avg_embeddings = {}
    for cls in unique_classes:
        if class_embeddings[cls]:
            avg_embedding = np.mean(np.array(class_embeddings[cls]), axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            avg_embeddings[cls] = avg_embedding.tolist()
    
    # Create output dictionary
    output_data = {
        "class_names": unique_classes,
        "text_embeddings": {cls: emb.tolist() for cls, emb in text_embeddings.items()},
        "avg_embeddings": avg_embeddings,
        "class_embeddings": class_embeddings,
        "class_paths": class_paths,
        "model": args.model
    }
    
    # Save embeddings to JSON file
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Embeddings saved to {args.output}")
    
    # Create a smaller version with just the average embeddings
    small_output = {
        "class_names": unique_classes,
        "avg_embeddings": avg_embeddings,
        "text_embeddings": {cls: emb.tolist() for cls, emb in text_embeddings.items()},
        "model": args.model
    }
    
    small_output_path = os.path.splitext(args.output)[0] + "_small.json"
    with open(small_output_path, 'w') as f:
        json.dump(small_output, f, indent=2)
    
    print(f"Small version saved to {small_output_path}")

if __name__ == "__main__":
    main()
