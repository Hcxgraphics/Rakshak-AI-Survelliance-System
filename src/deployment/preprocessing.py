from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_weapon(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess frame for YOLOv8 weapon model.
    BGR -> RGB, resize to 640x640, normalize [0, 1].
    Note: YOLOv8 model.predict() can handle raw numpy arrays, 
    but we provide the preprocessed RGB image explicitly.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Invalid frame for weapon preprocessing")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    return frame_rgb # YOLOv8 predict internally prefers RGB image format, return frame_rgb

def preprocess_violence(
    frames_bgr: list[np.ndarray], 
    sequence_length: int = 16, 
    target_size: tuple[int, int] = (64, 64)
) -> np.ndarray:
    """
    Preprocess sequence of frames for MobileNet+LSTM violence model.
    Resizes to target_size, applies ImageNet mean/std normalization, 
    and handles sequence batching for LSTM temporal input.
    """
    if not frames_bgr:
        raise ValueError("Violence model needs at least one frame.")
    
    # Sequence selection/interpolation
    if len(frames_bgr) >= sequence_length:
        indices = np.linspace(0, len(frames_bgr) - 1, num=sequence_length, dtype=int)
        selected = [frames_bgr[index] for index in indices]
    else:
        selected = [*frames_bgr]
        while len(selected) < sequence_length:
            selected.append(selected[-1].copy())
            
    processed_frames = []
    for frame in selected:
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame_resized = cv2.resize(frame_rgb, target_size)
        # Scale to [0, 1] and normalize with ImageNet stats
        frame_normalized = (frame_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        processed_frames.append(frame_normalized)
        
    clip = np.stack(processed_frames, axis=0)
    # Add batch dimension: (1, sequence_length, height, width, channels)
    return np.expand_dims(clip, axis=0)

def preprocess_police(
    frame_bgr: np.ndarray, 
    transform: transforms.Compose
) -> torch.Tensor:
    """
    Preprocess frame for MobileNetV2 police model.
    BGR -> RGB, resize to 224x224, normalize with ImageNet stats.
    Returns PyTorch tensor with batch dimension.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Invalid frame for police preprocessing")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor = transform(pil_img) # Resize + ToTensor + Normalize
    return tensor.unsqueeze(0) # (1, 3, 224, 224)

def preprocess_accident(
    frame_bgr: np.ndarray, 
    transform: transforms.Compose
) -> torch.Tensor:
    """
    Preprocess frame for ResNet50 accident model.
    BGR -> RGB, resize to 224x224, normalize with ImageNet stats.
    Returns PyTorch tensor with batch dimension.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Invalid frame for accident preprocessing")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor = transform(pil_img) # Resize + ToTensor + Normalize
    return tensor.unsqueeze(0) # (1, 3, 224, 224)
