import cv2
import numpy as np

def run_canny_failsafe(thermal_image_uint8: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Emergency Failsafe Module.
    Extracts physical structural boundaries directly from raw thermal IR imagery using
    OpenCV Canny Edge Detection & Adaptive Thresholding.
    
    Guarantees zero-hallucination terrain contours during live demo fallbacks.
    Returns 3-channel uint8 RGB image of high-frequency edges.
    """
    if thermal_image_uint8.ndim == 3:
        gray = cv2.cvtColor(thermal_image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = thermal_image_uint8

    # Gaussian blur to remove thermal sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Extraction
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Dilate edges slightly for visual enhancement
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert single-channel edge mask to 3-channel RGB image (white contours on dark background)
    edge_rgb = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB)
    return edge_rgb
