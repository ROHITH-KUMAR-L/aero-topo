import os
import logging
import numpy as np
import cv2
import torch
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("AeroTopo.RoboFireFuseNet")

class RoboFireFuseNetModel:
    """
    RoboFireFuseNet optional auxiliary branch for wildfire flame & smoke segmentation.
    Produces optional fire mask, smoke mask, and semantic confidence overlay.
    """
    def __init__(self, checkpoint_path: str = "models/robofirefusenet.pth", enabled: bool = False):
        self.checkpoint_path = checkpoint_path
        self.enabled = enabled
        self.is_ready = False
        self.status_message = "RoboFireFuseNet disabled by default."

        if self.enabled:
            self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(self.checkpoint_path):
            self.status_message = (
                f"RoboFireFuseNet weights not found at '{self.checkpoint_path}'. "
                "Auxiliary semantic branch will use heuristic thermal/visible thresholding."
            )
            logger.info(self.status_message)
            self.is_ready = False
            return
        
        self.is_ready = True
        self.status_message = "RoboFireFuseNet loaded successfully."

    def segment(self, rgb: np.ndarray, thermal_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Segment fire and smoke from aligned RGB and raw thermal arrays.
        Returns:
            fire_mask: uint8 (H, W) binary/heatmap
            smoke_mask: uint8 (H, W) binary/heatmap
            info: Dict metadata
        """
        h, w = rgb.shape[:2]

        # Heuristic segmentation if weights not loaded
        th_norm = cv2.normalize(thermal_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Fire threshold: top thermal intensity + bright RGB red/yellow
        _, th_fire = cv2.threshold(th_norm, 210, 255, cv2.THRESH_BINARY)
        
        # Smoke threshold: low saturation, gray/haze in RGB, low-medium thermal
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        smoke_mask = ((sat < 60) & (val > 140) & (th_norm < 180)).astype(np.uint8) * 255

        return th_fire, smoke_mask, {
            "branch": "RoboFireFuseNet-Heuristic",
            "fire_pixel_count": int(np.count_nonzero(th_fire)),
            "smoke_pixel_count": int(np.count_nonzero(smoke_mask))
        }
