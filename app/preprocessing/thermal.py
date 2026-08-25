import cv2
import numpy as np
from PIL import Image
import tifffile
from typing import Tuple, Dict, Any

def load_thermal_image(file_path_or_bytes) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load thermal image from file path or bytes.
    Supports 8-bit, 16-bit (radiometric TIFF), and multi-channel thermal formats.
    Returns:
        raw_thermal: float32 numpy array of raw thermal values (original precision preserved)
        normalized_thermal: uint8 3-channel image normalized to [0, 255] for visual/model processing
        metadata: Dict with original dtype, min/max values, shape info.
    """
    raw_thermal = None
    
    # Try reading with tifffile for 16-bit TIFF
    try:
        if isinstance(file_path_or_bytes, str):
            if file_path_or_bytes.lower().endswith(('.tif', '.tiff')):
                raw_thermal = tifffile.imread(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, bytes):
            try:
                raw_thermal = tifffile.imread(file_path_or_bytes)
            except Exception:
                pass
    except Exception:
        raw_thermal = None

    if raw_thermal is None:
        if isinstance(file_path_or_bytes, str):
            raw_thermal = cv2.imread(file_path_or_bytes, cv2.IMREAD_UNCHANGED)
        elif isinstance(file_path_or_bytes, bytes):
            nparr = np.frombuffer(file_path_or_bytes, np.uint8)
            raw_thermal = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if raw_thermal is None:
        raise ValueError("Failed to decode thermal image file.")

    original_dtype = str(raw_thermal.dtype)
    shape = raw_thermal.shape

    # Handle multi-channel thermal (convert to 1-channel raw thermal matrix if 3-channel)
    if len(raw_thermal.shape) == 3 and raw_thermal.shape[2] == 3:
        # Convert BGR/RGB to grayscale
        raw_thermal_single = cv2.cvtColor(raw_thermal, cv2.COLOR_BGR2GRAY)
    else:
        raw_thermal_single = raw_thermal

    raw_float = raw_thermal_single.astype(np.float32)
    min_val = float(np.min(raw_float))
    max_val = float(np.max(raw_float))

    # Min-max normalize to [0, 255] uint8 for visual/tensor representation
    val_range = max_val - min_val
    if val_range < 1e-6:
        norm_uint8 = np.zeros_like(raw_float, dtype=np.uint8)
    else:
        norm_uint8 = np.clip(((raw_float - min_val) / val_range) * 255.0, 0, 255).astype(np.uint8)

    # 3-channel uint8 representation for torch models expecting (3, H, W)
    norm_3ch = cv2.cvtColor(norm_uint8, cv2.COLOR_GRAY2RGB)

    metadata = {
        "original_dtype": original_dtype,
        "shape": shape,
        "min_value": min_val,
        "max_value": max_val,
        "is_16bit": "16" in original_dtype
    }

    return raw_float, norm_3ch, metadata
