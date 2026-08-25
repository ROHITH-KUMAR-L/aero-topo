import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def load_rgb_image(file_path_or_bytes) -> np.ndarray:
    """
    Load RGB image from file path or bytes.
    Returns RGB uint8 image array (H, W, 3).
    """
    if isinstance(file_path_or_bytes, str):
        img_bgr = cv2.imread(file_path_or_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image from {file_path_or_bytes}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(file_path_or_bytes, bytes):
        nparr = np.frombuffer(file_path_or_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode RGB image bytes.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(file_path_or_bytes, np.ndarray):
        img_rgb = file_path_or_bytes
    else:
        raise TypeError("Invalid image input type.")

    return img_rgb
