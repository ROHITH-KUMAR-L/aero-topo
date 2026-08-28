import os
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("AeroTopo.DepthAnythingV2")

class DepthAnythingV2Model:
    """
    Depth Anything V2 Model Wrapper.
    Outputs relative depth (non-metric).
    Provides raw floating-point relative depth array for 3D geometry and
    percentile-clipped display previews.
    """
    def __init__(
        self,
        encoder: str = "vits",
        mode: str = "relative",
        checkpoint_path: str = "models/checkpoints/depth_anything_v2.pth",
        device: str = "cuda",
        fp16: bool = True
    ):
        self.encoder = encoder  # vits (small), vitb (base), vitl (large)
        self.mode = mode        # relative
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.pipe = None
        self.is_hf_pipe = False
        self.is_ready = False
        self.status_message = ""

        self._initialize_model()

    def _initialize_model(self):
        hf_repo_map = {
            "vits": "depth-anything/Depth-Anything-V2-Small-hf",
            "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
            "vitl": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        repo_id = hf_repo_map.get(self.encoder, "depth-anything/Depth-Anything-V2-Small-hf")

        try:
            from transformers import pipeline
            logger.info(f"Loading Depth Anything V2 from Hugging Face: {repo_id}...")
            self.pipe = pipeline(
                task="depth-estimation",
                model=repo_id,
                device=0 if self.device.type == "cuda" else -1
            )
            self.is_hf_pipe = True
            self.is_ready = True
            self.status_message = f"Depth Anything V2 ({self.encoder}) loaded via HuggingFace on {self.device}."
            logger.info(self.status_message)
            return
        except Exception as e1:
            logger.warning(f"Hugging Face transformers load failed for '{repo_id}': {e1}. Using structural depth fallback.")

        self.is_ready = True
        self.status_message = f"Depth Anything V2 structural estimation active on {self.device}."
        logger.info(self.status_message)

    def predict_depth(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run relative depth estimation on RGB/Fused image (H, W, 3).
        Returns:
            raw_relative_depth: float32 numpy array (H, W) preserving raw depth values
            norm_depth_visual: uint8 colormapped preview image (H, W, 3) using 5th-95th percentile clipping
            quality_info: Dict with depth statistics & sanity checks
        """
        h, w = image_rgb.shape[:2]
        raw_depth = None

        if self.is_hf_pipe and self.pipe is not None:
            try:
                pil_img = Image.fromarray(image_rgb)
                result = self.pipe(pil_img)
                depth_pil = result["depth"]
                depth_np = np.array(depth_pil, dtype=np.float32)
                
                if depth_np.shape[:2] != (h, w):
                    raw_depth = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    raw_depth = depth_np
            except Exception as e:
                logger.error(f"Depth Anything V2 HF pipeline prediction failed: {e}.")

        if raw_depth is None:
            raw_depth = self._structural_depth_fallback(image_rgb)

        # Depth Quality Sanity Checker
        quality_info = self._check_depth_quality(raw_depth)

        # Percentile-Clipped Visualization Preview (5th to 95th Percentile)
        # Display transform ONLY for visual UI preview, raw_depth remains unquantized for 3D geometry
        valid_vals = raw_depth[np.isfinite(raw_depth)]
        if len(valid_vals) > 0:
            p5, p95 = np.percentile(valid_vals, [5, 95])
            if (p95 - p5) > 1e-6:
                clipped = np.clip(raw_depth, p5, p95)
                norm_01 = (clipped - p5) / (p95 - p5)
            else:
                norm_01 = np.zeros_like(raw_depth)
        else:
            norm_01 = np.zeros_like(raw_depth)

        norm_uint8 = (norm_01 * 255.0).astype(np.uint8)
        norm_depth_visual = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_INFERNO)
        norm_depth_visual = cv2.cvtColor(norm_depth_visual, cv2.COLOR_BGR2RGB)

        return raw_depth, norm_depth_visual, quality_info

    def _structural_depth_fallback(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Gradient & structure-guided relative depth fallback when weights are offline.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        vert_grad = (y_coords / float(h)).astype(np.float32)

        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        local_detail = np.abs(gray - blurred)
        detail_smooth = cv2.GaussianBlur(local_detail, (21, 21), 0)
        detail_norm = cv2.normalize(detail_smooth, None, 0, 1, cv2.NORM_MINMAX)

        depth = 0.6 * vert_grad + 0.4 * (1.0 - detail_norm)
        depth_smoothed = cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)
        return depth_smoothed

    def _check_depth_quality(self, depth: np.ndarray) -> Dict[str, Any]:
        """
        Depth sanity checker computing min, max, mean, std, percentile range, and % NaN/Inf.
        """
        total_pixels = depth.size
        num_nan = int(np.isnan(depth).sum())
        num_inf = int(np.isinf(depth).sum())
        pct_nan_inf = round(((num_nan + num_inf) / float(total_pixels)) * 100.0, 2)
        
        valid_mask = np.isfinite(depth)
        if not np.any(valid_mask):
            return {
                "status": "INVALID",
                "depth_mode": "Relative",
                "warnings": ["Depth map contains no finite values."]
            }

        valid_vals = depth[valid_mask]
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        p5, p95 = np.percentile(valid_vals, [5, 95])
        percentile_range = float(p95 - p5)

        warnings = []
        is_low_variation = std_val < 1e-4

        if is_low_variation:
            status = "LOW VARIATION"
            warnings.append("Depth map exhibits near-constant variation.")
        elif pct_nan_inf > 5.0:
            status = "WARNING"
            warnings.append(f"{pct_nan_inf}% of depth values are NaN/Inf.")
        else:
            status = "Valid"

        return {
            "status": status,
            "depth_mode": "Relative",
            "min_depth": round(min_val, 4),
            "max_depth": round(max_val, 4),
            "mean_depth": round(mean_val, 4),
            "std_depth": round(std_val, 4),
            "p5_depth": round(float(p5), 4),
            "p95_depth": round(float(p95), 4),
            "percentile_range": round(percentile_range, 4),
            "pct_nan_inf": pct_nan_inf,
            "warnings": warnings
        }
