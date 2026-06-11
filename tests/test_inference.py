from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure src/deployment is on the python search path
DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "src" / "deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

import preprocessing
import fusion_engine
from models import load_models
from inference import (
    detect_objects,
    _detect_weapon_wrapped,
    police_detect_wrapped,
    _predict_violence_score_wrapped,
    _predict_accident_wrapped,
)

def test_model_loading(models):
    """Verify that all models load correctly and have correct types."""
    print("Running test_model_loading...")
    assert models.weapon_model is not None
    assert models.violence_model is not None
    assert models.police_head is not None
    assert models.police_backbone is not None
    assert models.accident_model is not None
    print("test_model_loading PASSED")

def test_preprocessing_helpers(models):
    """Verify that preprocessing functions return correct shapes and types."""
    print("Running test_preprocessing_helpers...")
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # 1. Weapon preprocessor
    weapon_prep = preprocessing.preprocess_weapon(dummy_bgr)
    assert isinstance(weapon_prep, np.ndarray)
    assert weapon_prep.shape == (480, 640, 3) # Returns RGB frame

    # 2. Violence preprocessor
    dummy_seq = [dummy_bgr] * 5
    violence_prep = preprocessing.preprocess_violence(dummy_seq, sequence_length=16, target_size=(64, 64))
    assert isinstance(violence_prep, np.ndarray)
    assert violence_prep.shape == (1, 16, 64, 64, 3)

    # 3. Police preprocessor
    police_prep = preprocessing.preprocess_police(dummy_bgr, models.police_transform)
    assert isinstance(police_prep, torch.Tensor)
    assert police_prep.shape == (1, 3, 224, 224)

    # 4. Accident preprocessor
    accident_prep = preprocessing.preprocess_accident(dummy_bgr, models.transform_accident)
    assert isinstance(accident_prep, torch.Tensor)
    assert accident_prep.shape == (1, 3, 224, 224)
    print("test_preprocessing_helpers PASSED")

def test_weapon_wrapper(models):
    """Test the weapon detection wrapper using a dummy frame."""
    print("Running test_weapon_wrapper...")
    dummy_rgb = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    annotated = dummy_rgb.copy()
    
    res = _detect_weapon_wrapped(dummy_rgb, annotated, threshold=0.5)
    assert "weapon_detections" in res
    assert "weapons_detected" in res
    assert "has_weapon_yolo" in res
    assert "weapon_score_yolo" in res
    assert "standardized" in res
    
    std = res["standardized"]
    assert "class" in std
    assert "confidence" in std
    assert "boxes" in std
    assert isinstance(std["class"], str)
    assert isinstance(std["confidence"], float)
    print("test_weapon_wrapper PASSED")

def test_police_wrapper(models):
    """Test the police detection wrapper using a dummy frame."""
    print("Running test_police_wrapper...")
    dummy_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    res = police_detect_wrapped(dummy_bgr, threshold=0.35)
    assert "has_police" in res
    assert "police_score" in res
    assert "police_scores" in res
    assert "standardized" in res
    
    std = res["standardized"]
    assert std["class"] in {"police", "normal"}
    assert isinstance(std["confidence"], float)
    print("test_police_wrapper PASSED")

def test_violence_wrapper(models):
    """Test the violence prediction wrapper using a dummy sequence."""
    print("Running test_violence_wrapper...")
    dummy_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    dummy_input = np.random.randn(1, 16, 64, 64, 3).astype(np.float32)
    
    res = _predict_violence_score_wrapped(dummy_rgb, dummy_input)
    assert "violence_score_lstm" in res
    assert "standardized" in res
    
    std = res["standardized"]
    assert std["class"] in {"violence", "normal"}
    assert isinstance(std["confidence"], float)
    print("test_violence_wrapper PASSED")

def test_accident_wrapper(models):
    """Test the accident detection wrapper using a dummy frame."""
    print("Running test_accident_wrapper...")
    dummy_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    res = _predict_accident_wrapped(dummy_bgr)
    assert "accident_class" in res
    assert "accident_conf" in res
    assert "standardized" in res
    
    std = res["standardized"]
    assert isinstance(std["confidence"], float)
    print("test_accident_wrapper PASSED")

def test_full_fusion_pipeline(models):
    """Verify that detect_objects correctly runs full fusion and returns expected keys."""
    print("Running test_full_fusion_pipeline...")
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    res = detect_objects(
        dummy_frame,
        is_video=False,
        threshold_weapon=0.5,
        threshold_violence=0.6,
        threshold_accident=0.7,
        threshold_police=0.35,
        detection_enabled=True,
    )
    
    expected_keys = [
        "is_video", "source_type", "title", "summary", "scene", "risk_level",
        "fusion_confidence", "weapons_detected", "weapon_detections",
        "has_weapon_yolo", "weapon_score_yolo", "gun_score_police",
        "knife_score_police", "has_police", "police_score", "police_top_label",
        "police_top_score", "police_scores", "violence_score_lstm",
        "violence_score_police", "accident_class", "accident_conf",
        "component_errors", "component_latency_ms", "signals", "image"
    ]
    
    for key in expected_keys:
        assert key in res, f"Expected key '{key}' not found in detect_objects output."
        
    assert isinstance(res["image"], bytes)
    assert res["risk_level"] in {"critical", "high", "watch", "low"}
    assert isinstance(res["fusion_confidence"], float)
    assert isinstance(res["signals"], list)
    print("test_full_fusion_pipeline PASSED")

if __name__ == "__main__":
    print("Loading models...")
    m = load_models()
    print("Models loaded successfully!")
    
    test_model_loading(m)
    test_preprocessing_helpers(m)
    test_weapon_wrapper(m)
    test_police_wrapper(m)
    test_violence_wrapper(m)
    test_accident_wrapper(m)
    test_full_fusion_pipeline(m)
    
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
