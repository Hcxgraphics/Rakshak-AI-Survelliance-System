#!/usr/bin/env python3
"""Test script to run detection on sample images from sampleTestingImages folder."""

import json
import sys
from pathlib import Path
from time import sleep

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
API_URL = "http://127.0.0.1:8001"
SAMPLE_IMAGES_DIR = Path(__file__).parent.parent / "sampleTestingImages"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}

def setup_session():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def check_api_health(session):
    """Check if the API is running and models are loaded."""
    try:
        response = session.get(f"{API_URL}/health", timeout=5)
        result = response.json()
        print(f"\n✓ API Health Check: {result['status']}")
        if result.get('models_loaded'):
            print("✓ Models loaded successfully")
            for model_name, model_path in result.get('models', {}).items():
                if model_name != 'device':
                    print(f"  • {model_name}: {Path(model_path).name}")
        else:
            print("✗ Models not loaded - waiting for backend to initialize...")
            return False
        return True
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        print(f"  Make sure the backend is running on {API_URL}")
        return False

def get_sample_files():
    """Get list of sample media files."""
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"✗ Sample images directory not found: {SAMPLE_IMAGES_DIR}")
        return []
    
    files = sorted([f for f in SAMPLE_IMAGES_DIR.iterdir() 
                   if f.suffix.lower() in SUPPORTED_FORMATS])
    return files

def upload_and_detect(session, file_path, threshold=0.55):
    """Upload a file and run detection."""
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return None
    
    print(f"\n📤 Processing: {file_path.name}")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'threshold': threshold,
                'save_evidence': True,
                'mode': 'auto'
            }
            response = session.post(
                f"{API_URL}/upload",
                files=files,
                data=data,
                timeout=120
            )
        
        if response.status_code == 200:
            result = response.json()
            display_detection_results(result, file_path)
            return result
        else:
            print(f"✗ Detection failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return None

def display_detection_results(result, file_path):
    """Display formatted detection results."""
    print(f"\n📊 Detection Results for {file_path.name}:")
    print(f"  Request ID: {result.get('request_id', 'N/A')}")
    
    # Scene and risk assessment
    print(f"\n  Scene Type: {result.get('scene', 'Normal')}")
    print(f"  Risk Level: {result.get('risk_level', 'low').upper()}")
    print(f"  Fusion Confidence: {result.get('fusion_confidence', 0):.2%}")
    
    # Police Detection
    if result.get('police_detected'):
        print(f"\n  🚔 Police Detected:")
        print(f"     Confidence: {result.get('police_score', 0):.2%}")
    
    # Weapon Detection
    if result.get('weapon_count', 0) > 0:
        print(f"\n  🔫 Weapons Detected: {result.get('weapon_count')}")
        print(f"     Gun Score: {result.get('gun_score_police', 0):.2%}")
        print(f"     Knife Score: {result.get('knife_score_police', 0):.2%}")
        if result.get('weapon_detections'):
            for det in result['weapon_detections'][:3]:  # Show first 3
                print(f"     • {det.get('class', 'Unknown')}")
    
    # Violence Detection
    violence_score = max(
        result.get('violence_score_police', 0),
        result.get('violence_score_lstm', 0)
    )
    if violence_score > 0.3:
        print(f"\n  ⚠️  Violence Detected:")
        print(f"     Police Model: {result.get('violence_score_police', 0):.2%}")
        print(f"     LSTM Model: {result.get('violence_score_lstm', 0):.2%}")
    
    # Accident Detection
    if result.get('accident_class') is not None:
        accident_classes = {0: "Normal", 1: "Accident", 2: "Dangerous"}
        class_name = accident_classes.get(result['accident_class'], "Unknown")
        print(f"\n  🚗 Accident Assessment: {class_name}")
        print(f"     Confidence: {result.get('accident_confidence', 0):.2%}")
    
    # Component latencies
    if result.get('component_latency_ms'):
        print(f"\n  ⏱️  Processing Times:")
        for component, latency in result['component_latency_ms'].items():
            print(f"     {component}: {latency:.0f}ms")
    
    # Errors if any
    if result.get('component_errors'):
        print(f"\n  ⚠️  Component Errors:")
        for component, error in result['component_errors'].items():
            print(f"     {component}: {error}")
    
    if result.get('saved_path'):
        print(f"\n  💾 Evidence saved: {result['saved_path']}")

def main():
    """Main execution."""
    print("\n" + "="*60)
    print("🎥 AI-Powered Public Safety Surveillance System Test")
    print("="*60)
    
    session = setup_session()
    
    # Wait for API to be ready
    print("\n⏳ Connecting to API backend...")
    max_retries = 6
    for attempt in range(max_retries):
        if check_api_health(session):
            break
        if attempt < max_retries - 1:
            print(f"   Retrying in 5 seconds... ({attempt + 1}/{max_retries - 1})")
            sleep(5)
    else:
        print("\n✗ Failed to connect to API after multiple attempts")
        print("   Please ensure the backend is running:")
        print(f"   cd AI-Powered-Public-Safety-Surveillance-System")
        print(f"   python -m uvicorn src.deployment.main:app --reload")
        return 1
    
    # Get sample files
    sample_files = get_sample_files()
    if not sample_files:
        print(f"\n✗ No supported media files found in {SAMPLE_IMAGES_DIR}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return 1
    
    print(f"\n📁 Found {len(sample_files)} sample files:")
    for i, f in enumerate(sample_files, 1):
        print(f"   {i}. {f.name}")
    
    # Process all samples
    print("\n" + "-"*60)
    results_summary = []
    for file_path in sample_files:
        result = upload_and_detect(session, file_path, threshold=0.55)
        if result:
            results_summary.append({
                'file': file_path.name,
                'scene': result.get('scene', 'Normal'),
                'risk_level': result.get('risk_level', 'low'),
                'weapons': result.get('weapon_count', 0),
                'police': result.get('police_detected', False),
                'violence': result.get('risk_level', 'low') in ['medium', 'high'],
            })
    
    # Summary report
    print("\n" + "="*60)
    print("📋 Summary Report")
    print("="*60)
    print(f"Total files processed: {len(results_summary)}")
    
    if results_summary:
        print("\nDetection Summary:")
        for summary in results_summary:
            indicators = []
            if summary['weapons'] > 0:
                indicators.append(f"🔫 {summary['weapons']} weapon(s)")
            if summary['police']:
                indicators.append("🚔 Police")
            if summary['violence']:
                indicators.append("⚠️ Violence")
            
            indicator_str = " | ".join(indicators) if indicators else "✓ Normal"
            print(f"  • {summary['file']}: {indicator_str} (Risk: {summary['risk_level'].upper()})")
    
    print("\n✓ Testing complete!")
    print("="*60 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
