import cv2
import numpy as np
from typing import Tuple, Dict, Any

def align_image_pair(
    rgb: np.ndarray,
    thermal_3ch: np.ndarray,
    raw_thermal: np.ndarray,
    target_size: Tuple[int, int] = (640, 512)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Spatially align RGB and thermal imagery to match target dimensions.
    Applies identical geometric transforms to both modalities to preserve alignment.
    """
    h_rgb, w_rgb = rgb.shape[:2]
    h_th, w_th = thermal_3ch.shape[:2]

    mismatch_detected = (h_rgb != h_th) or (w_rgb != w_th)
    
    target_w, target_h = target_size

    rgb_aligned = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    thermal_3ch_aligned = cv2.resize(thermal_3ch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    raw_thermal_aligned = cv2.resize(raw_thermal, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    alignment_info = {
        "mismatch_detected": mismatch_detected,
        "original_rgb_size": [w_rgb, h_rgb],
        "original_thermal_size": [w_th, h_th],
        "aligned_size": [target_w, target_h]
    }

    return rgb_aligned, thermal_3ch_aligned, raw_thermal_aligned, alignment_info

def compute_smoke_confidence(rgb: np.ndarray) -> Dict[str, Any]:
    """
    Calculate qualitative smoke interference level based on visual degradation metrics:
    - Low contrast (std dev of grayscale intensity)
    - Low saturation (HSV saturation channel)
    - Low high-frequency edge density (Laplacian variance)
    - Near-white/gray haze concentration
    
    NOTE: Returns qualitative level ("Low", "Medium", "High") per scientific honesty requirements.
    No pseudo-precise numeric percentages are claimed for heuristic estimates.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    sat_mean = float(np.mean(hsv[:, :, 1]) / 255.0)
    std_dev = float(np.std(gray) / 128.0)
    
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_density = float(np.clip(lap_var / 500.0, 0.0, 1.0))

    brightness = gray.astype(np.float32) / 255.0
    haze_pixels = (brightness > 0.6) & (hsv[:, :, 1] < 50)
    haze_ratio = float(np.mean(haze_pixels))

    smoke_score = float(np.clip((1.0 - sat_mean * 0.5 - edge_density * 0.5 + haze_ratio * 0.5), 0.0, 1.0))

    if smoke_score < 0.35:
        smoke_level = "Low"
        visibility_level = "High"
    elif smoke_score < 0.65:
        smoke_level = "Medium"
        visibility_level = "Medium"
    else:
        smoke_level = "High"
        visibility_level = "Low"

    return {
        "estimate_type": "Heuristic",
        "smoke_level": smoke_level,
        "visibility_level": visibility_level,
        "description": f"Heuristic smoke estimation: {smoke_level} interference"
    }
