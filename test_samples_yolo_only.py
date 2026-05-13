#!/usr/bin/env python3
"""Simplified test script using only YOLO models that work with current dependencies."""

import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Configuration - use absolute path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SAMPLE_IMAGES_DIR = PROJECT_ROOT.parent / "sampleTestingImages"
MODELS_DIR = PROJECT_ROOT / "models"

def test_image_with_yolo(image_path: Path, yolo_model: YOLO) -> dict:
    """Run detection on an image using YOLO."""
    if not image_path.exists():
        print(f"✗ File not found: {image_path}")
        return {}
    
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"✗ Could not load image: {image_path}")
        return {}
    
    height, width = frame.shape[:2]
    print(f"\n📤 Processing: {image_path.name} ({width}x{height})")
    
    # Run YOLO detection
    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    result = results[0]
    
    # Extract detections
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy.tolist()[0] if len(box.xyxy.tolist()) > 0 else []
            })
    
    return {
        'file': image_path.name,
        'detections': detections,
        'num_objects': len(detections),
        'image_size': (width, height),
    }

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("🔍 AI Surveillance System - Sample Image Testing")
    print("="*70)
    
    # Get sample files
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"\n✗ Sample images directory not found: {SAMPLE_IMAGES_DIR}")
        return 1
    
    sample_files = sorted([f for f in SAMPLE_IMAGES_DIR.iterdir() 
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
    
    if not sample_files:
        print(f"✗ No image files found in {SAMPLE_IMAGES_DIR}")
        return 1
    
    print(f"\n📁 Found {len(sample_files)} sample images:")
    for i, f in enumerate(sample_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {i}. {f.name} ({size_mb:.1f} MB)" if size_mb > 1 else f"   {i}. {f.name}")
    
    # Load YOLO weapon model
    print("\n⏳ Loading YOLO weapon detection model...")
    weapon_model_path = MODELS_DIR / "weapon" / "best.pt"
    
    if not weapon_model_path.exists():
        print(f"✗ Weapon model not found: {weapon_model_path}")
        return 1
    
    try:
        weapon_model = YOLO(str(weapon_model_path))
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return 1
    
    # Process all samples
    print("\n" + "-"*70)
    print("Running Detection...")
    print("-"*70)
    
    all_results = []
    for file_path in sample_files:
        result = test_image_with_yolo(file_path, weapon_model)
        if result:
            all_results.append(result)
            
            # Display results
            if result['detections']:
                print(f"  ✓ {result['num_objects']} object(s) detected:")
                for det in result['detections']:
                    confidence_pct = det['confidence'] * 100
                    print(f"     • {det['class']}: {confidence_pct:.1f}%")
            else:
                print(f"  ✓ No weapons detected (normal scene)")
    
    # Summary
    print("\n" + "="*70)
    print("📋 Summary Report")
    print("="*70)
    
    total_file = len(all_results)
    total_detections = sum(r['num_objects'] for r in all_results)
    
    print(f"Total files processed: {total_file}")
    print(f"Total objects detected: {total_detections}")
    
    if all_results:
        print(f"\n📊 Detection Details:")
        for result in all_results:
            status = f"⚠️  {result['num_objects']} threats" if result['detections'] else "✓ Normal"
            print(f"  • {result['file']}: {status}")
    
    print("\n✓ Testing complete!")
    print("="*70 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
