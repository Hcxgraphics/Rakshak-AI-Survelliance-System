from __future__ import annotations

from enum import Enum
from typing import Any

class ThreatLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SAFE = "SAFE"

def standardize_model_output(
    class_name: str, 
    confidence: float, 
    boxes: list[list[int]] | None = None
) -> dict[str, Any]:
    """
    Standardize each model's output to the required format:
    {"class": str, "confidence": float, "boxes": list or None}
    """
    return {
        "class": str(class_name),
        "confidence": float(confidence),
        "boxes": boxes if boxes is not None else None
    }

def fuse_detections(
    weapon_out: dict[str, Any],
    violence_out: dict[str, Any],
    accident_out: dict[str, Any],
    police_out: dict[str, Any],
    thresholds: dict[str, float]
) -> dict[str, Any]:
    """
    Ingests the four standardized outputs and runs fusion priority logic:
    Weapon > Violence > Accident > Police.
    Returns:
        {
            "threat_level": ThreatLevel,
            "risk_level": str,  # Lowercase for frontend styling compatibility
            "scene": str,
            "summary": str,
            "confidence": float
        }
    """
    # Check activation for each model based on thresholds
    weapon_active = (
        weapon_out["class"] != "normal" and 
        weapon_out["confidence"] >= thresholds.get("weapon", 0.5)
    )
    violence_active = (
        violence_out["class"] == "violence" and 
        violence_out["confidence"] >= thresholds.get("violence", 0.6)
    )
    accident_active = (
        accident_out["class"] != "normal" and 
        accident_out["confidence"] >= thresholds.get("accident", 0.7)
    )
    police_active = (
        police_out["class"] == "police" and 
        police_out["confidence"] >= thresholds.get("police", 0.35)
    )

    # 1. Weapon priority -> CRITICAL
    if weapon_active:
        summary = f"Weapon detected: {weapon_out['class']}"
        return {
            "threat_level": ThreatLevel.CRITICAL,
            "risk_level": "critical",
            "scene": "Weapon",
            "summary": summary,
            "confidence": weapon_out["confidence"]
        }

    # 2. Violence priority -> HIGH
    if violence_active:
        # Check if police is also present during violence (hybrid detection)
        if police_active:
            summary = "Police present with violence detected"
        else:
            summary = "Violence detected"
        return {
            "threat_level": ThreatLevel.HIGH,
            "risk_level": "high",
            "scene": "Violence",
            "summary": summary,
            "confidence": violence_out["confidence"]
        }

    # 3. Accident priority -> HIGH/MEDIUM
    if accident_active:
        summary = f"Accident detected ({accident_out['class']})"
        return {
            "threat_level": ThreatLevel.HIGH,
            "risk_level": "high",
            "scene": "Accident",
            "summary": summary,
            "confidence": accident_out["confidence"]
        }

    # 4. Police priority -> MEDIUM/LOW (Watch status)
    if police_active:
        summary = "Police present"
        return {
            "threat_level": ThreatLevel.MEDIUM,
            "risk_level": "watch",
            "scene": "Police",
            "summary": summary,
            "confidence": police_out["confidence"]
        }

    # Default -> SAFE
    # Find the maximum background signal to report confidence
    max_conf = max(
        weapon_out["confidence"],
        violence_out["confidence"],
        accident_out["confidence"],
        police_out["confidence"],
        0.0
    )
    return {
        "threat_level": ThreatLevel.SAFE,
        "risk_level": "low",
        "scene": "Normal",
        "summary": "Normal scene",
        "confidence": round(1.0 - max_conf, 4)
    }
