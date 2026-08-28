import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import requests
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("AeroTopo.FFFusion")

OFFICIAL_SOURCE_URL = "https://github.com/FF-Fusion/FF-Fusion"

class FFFusionStudentNet(nn.Module):
    """
    Lightweight Student Architecture inspired by FF-Fusion knowledge distillation.
    Input: RGB (3 channels) + Thermal (3 channels or 1 channel)
    Output: Fused visual representation (3 channels)
    """
    def __init__(self):
        super().__init__()
        # Dual-branch feature encoder
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.ir_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.fusion_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fusion_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        feat_rgb = self.relu(self.rgb_conv1(rgb))
        feat_ir = self.relu(self.ir_conv1(ir))
        
        cat_feat = torch.cat([feat_rgb, feat_ir], dim=1)
        fused = self.relu(self.fusion_conv1(cat_feat))
        fused = self.relu(self.fusion_conv2(fused))
        out = torch.sigmoid(self.fusion_out(fused))
        return out

class FFFusionModel:
    """
    FF-Fusion Model Wrapper.
    Handles weight loading and inference for visible-infrared multimodal fusion.
    """
    def __init__(self, checkpoint_path: str = "models/checkpoints/ff_fusion.pth", device: str = "cuda", fp16: bool = True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model: Optional[nn.Module] = None
        self.is_ready = False
        self.status_message = ""

        self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(self.checkpoint_path):
            self.status_message = (
                f"FF-Fusion model weights not found at '{self.checkpoint_path}'. "
                f"Official source: {OFFICIAL_SOURCE_URL}. "
                "Place ff_fusion.pth in models/checkpoints/ or configure its Hugging Face repository."
            )
            logger.warning(self.status_message)
            self.is_ready = False
            return

        try:
            model = FFFusionStudentNet()
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.is_ready = True
            self.status_message = f"FF-Fusion model loaded successfully on {self.device}."
            logger.info(self.status_message)
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to load FF-Fusion weights from '{self.checkpoint_path}': {e}"
            logger.error(self.status_message)

    def fuse(self, rgb: np.ndarray, thermal_3ch: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Run fusion on paired Generated RGB (H, W, 3) and thermal_3ch (H, W, 3) uint8 numpy arrays.
        Returns:
            fused_image: uint8 numpy array (H, W, 3)
            info: Dict metadata
        """
        if not self.is_ready or self.model is None:
            return None, {
                "fusion_method": "FF-Fusion",
                "status": "UNAVAILABLE",
                "message": self.status_message,
                "label": "FF-Fusion Model Unavailable"
            }

        try:
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            ir_t = torch.from_numpy(thermal_3ch.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

            rgb_t = rgb_t.to(self.device)
            ir_t = ir_t.to(self.device)

            with torch.inference_mode():
                out_t = self.model(rgb_t, ir_t)

            fused_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            return fused_np, {
                "fusion_method": "FF-Fusion-Neural-Network",
                "status": "SUCCESS",
                "device": str(self.device)
            }
        except Exception as e:
            logger.error(f"FF-Fusion inference error: {e}")
            return None, {
                "fusion_method": "FF-Fusion",
                "status": "ERROR",
                "message": str(e),
                "label": "FF-Fusion Inference Error"
            }

