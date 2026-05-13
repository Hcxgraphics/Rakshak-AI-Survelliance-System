#!/usr/bin/env python3
"""Enhanced test script with detailed analysis and visual output."""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SAMPLE_IMAGES_DIR = PROJECT_ROOT.parent / "sampleTestingImages"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "test_results"


def analyze_image_content(image_path: Path) -> dict:
    """Analyze image properties and content."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {}
    
    height, width = frame.shape[:2]
    
    # Basic histogram analysis for scene intensity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    return {
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}",
        'aspect_ratio': round(width / height, 2),
        'pixel_count': width * height,
        'mean_intensity': round(mean_intensity, 2),
        'std_intensity': round(std_intensity, 2),
        'brightness': 'High' if mean_intensity > 150 else 'Medium' if mean_intensity > 100 else 'Low',
        'file_size_mb': round((image_path.stat().st_size) / (1024 * 1024), 2)
    }


def run_yolo_detection(image_path: Path, yolo_model: YOLO) -> dict:
    """Run YOLO detection on an image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {'error': 'Could not load image'}
    
    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    result = results[0]
    
    detections = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            det_dict = {
                'id': i + 1,
                'class': result.names.get(int(box.cls), 'Unknown'),
                'confidence': round(float(box.conf), 3),
                'bbox': [round(x, 1) for x in box.xyxy.tolist()[0]] if len(box.xyxy.tolist()) > 0 else [],
            }
            detections.append(det_dict)
    
    return {
        'model': 'YOLOv8',
        'num_detections': len(detections),
        'detections': detections,
        'inference_time': round(float(result.speed['inference']), 2) if hasattr(result, 'speed') else 0,
    }


def generate_report(all_results: list) -> str:
    """Generate a detailed text report."""
    report = []
    report.append("\n" + "="*80)
    report.append("📊 AI-POWERED PUBLIC SAFETY SURVEILLANCE SYSTEM")
    report.append("Sample Image Test Report")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Statistics
    total_files = len(all_results)
    files_with_detections = sum(1 for r in all_results if r['yolo']['num_detections'] > 0)
    total_detections = sum(r['yolo']['num_detections'] for r in all_results)
    
    report.append(f"\n{'─'*80}")
    report.append("SUMMARY STATISTICS")
    report.append(f"{'─'*80}")
    report.append(f"Total Images Processed:        {total_files}")
    report.append(f"Images with Detections:        {files_with_detections} ({100*files_with_detections/total_files:.1f}%)")
    report.append(f"Total Objects Detected:        {total_detections}")
    report.append(f"Average Objects per Image:     {total_detections/total_files:.2f}")
    
    total_inference_time = sum(r['yolo'].get('inference_time', 0) for r in all_results)
    report.append(f"Total Inference Time:          {total_inference_time:.2f} ms")
    report.append(f"Average Time per Image:        {total_inference_time/total_files:.2f} ms")
    
    # Threat Assessment
    report.append(f"\n{'─'*80}")
    report.append("THREAT ASSESSMENT")
    report.append(f"{'─'*80}")
    threat_level = "🟢 LOW" if total_detections == 0 else "🟡 MEDIUM" if total_detections <= 2 else "🔴 HIGH"
    report.append(f"Overall Threat Level:          {threat_level}")
    report.append(f"Risk Score (0-100):            {min(100, total_detections * 15)}")
    
    # Detailed Results
    report.append(f"\n{'─'*80}")
    report.append("DETAILED ANALYSIS")
    report.append(f"{'─'*80}\n")
    
    for r in all_results:
        report.append(f"📷 File: {r['filename']}")
        report.append(f"   Size: {r['image_info']['resolution']} | {r['image_info']['file_size_mb']} MB")
        report.append(f"   Brightness: {r['image_info']['brightness']} | Intensity: {r['image_info']['mean_intensity']}")
        
        if r['yolo']['num_detections'] > 0:
            report.append(f"   🎯 THREAT DETECTED: {r['yolo']['num_detections']} object(s)")
            for det in r['yolo']['detections']:
                conf_pct = det['confidence'] * 100
                severity = "⚠️ " if conf_pct > 80 else "ℹ️ " if conf_pct > 60 else "✓ "
                report.append(f"      {severity} {det['class']} ({conf_pct:.0f}% confidence)")
        else:
            report.append(f"   ✅ Scene appears normal - no threats detected")
        report.append("")
    
    # Recommendations
    report.append(f"{'─'*80}")
    report.append("RECOMMENDATIONS")
    report.append(f"{'─'*80}")
    if files_with_detections == 0:
        report.append("✓ No threats detected in any sample. System operating normally.")
    else:
        report.append("⚠️ Threats detected. Consider:")
        report.append("  • Manual review of detected scenes")
        report.append("  • Alerting security personnel")
        report.append("  • Check camera feeds for suspicious activity")
    
    report.append(f"\n{'='*80}\n")
    
    return "\n".join(report)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🔍 AI SURVEILLANCE SYSTEM - SAMPLE IMAGE TEST")
    print("="*80)
    
    # Get sample files
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"✗ Sample images directory not found: {SAMPLE_IMAGES_DIR}")
        return 1
    
    sample_files = sorted([f for f in SAMPLE_IMAGES_DIR.iterdir() 
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
    
    if not sample_files:
        print(f"✗ No image files found in {SAMPLE_IMAGES_DIR}")
        return 1
    
    print(f"\n📂 Sample Images Found: {len(sample_files)}")
    for i, f in enumerate(sample_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb > 1 else f"{f.stat().st_size / 1024:.0f} KB"
        print(f"   {i}. {f.name:<30} ({size_str})")
    
    # Load model
    print("\n⏳ Loading YOLOv8 Weapon Detection Model...")
    weapon_model_path = MODELS_DIR / "weapon" / "best.pt"
    
    if not weapon_model_path.exists():
        print(f"✗ Model not found: {weapon_model_path}")
        return 1
    
    try:
        weapon_model = YOLO(str(weapon_model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1
    
    # Processing
    print("\n" + "─"*80)
    print("Processing Images...")
    print("─"*80)
    
    all_results = []
    for idx, file_path in enumerate(sample_files, 1):
        print(f"\n[{idx}/{len(sample_files)}] {file_path.name}...", end=" ", flush=True)
        
        # Analyze image
        image_info = analyze_image_content(file_path)
        if not image_info:
            print("❌ Failed to load")
            continue
        
        # Run detection
        yolo_result = run_yolo_detection(file_path, weapon_model)
        if 'error' in yolo_result:
            print(f"❌ Detection failed: {yolo_result['error']}")
            continue
        
        result = {
            'filename': file_path.name,
            'image_info': image_info,
            'yolo': yolo_result,
        }
        all_results.append(result)
        
        # Status indicator
        if yolo_result['num_detections'] > 0:
            print(f"⚠️  {yolo_result['num_detections']} object(s) detected")
        else:
            print("✓ Normal")
    
    # Generate and display report
    report = generate_report(all_results)
    print(report)
    
    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON results
    json_file = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"💾 Report saved: {report_file.relative_to(PROJECT_ROOT)}")
    print(f"💾 JSON results: {json_file.relative_to(PROJECT_ROOT)}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
