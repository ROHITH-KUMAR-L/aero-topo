import os
from pathlib import Path
import numpy as np
import cv2
import torch

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_DEPTH_WEIGHTS = BASE_DIR / "weights" / "depth_anything_v2"
HF_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

class DepthEstimationEngine:
    def __init__(self, model_dir=LOCAL_DEPTH_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.processor = None
        self.model = None
        self.is_weights_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load Depth Anything V2 from local folder or HF repository."""
        if not HAS_TRANSFORMERS:
            print("[DepthEngine] Transformers library not present. Using gradient depth estimation.")
            return

        model_path = None
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            model_path = str(self.model_dir)
        else:
            model_path = HF_DEPTH_MODEL_ID

        print(f"[DepthEngine] Loading Depth Anything V2 from '{model_path}' on {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.is_weights_loaded = True
            print("[DepthEngine] Depth Anything V2 model loaded successfully!")
        except Exception as e:
            print(f"[DepthEngine] Could not load model from '{model_path}': {e}. Using elevation gradient fallback.")

    def estimate_depth(self, rgb_image_uint8: np.ndarray) -> np.ndarray:
        """
        Estimates relative depth matrix from 3-channel RGB image.
        Returns a normalized 2D uint8 numpy array [0, 255] representing the Z-map.
        """
        if self.is_weights_loaded and self.model is not None and self.processor is not None:
            try:
                # Prepare inputs for Hugging Face model
                from PIL import Image
                pil_img = Image.fromarray(rgb_image_uint8)
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    
                # Interpolate depth map to original image size
                h, w = rgb_image_uint8.shape[:2]
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                depth_np = prediction.cpu().numpy()
                
                # Normalize depth map to [0, 255]
                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max > d_min:
                    depth_uint8 = ((depth_np - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                else:
                    depth_uint8 = np.zeros((h, w), dtype=np.uint8)
                    
                return depth_uint8
            except Exception as e:
                print(f"[DepthEngine] Inference exception: {e}, falling back to gradient depth map.")

        return self._gradient_depth_fallback(rgb_image_uint8)

    def _gradient_depth_fallback(self, image_np: np.ndarray) -> np.ndarray:
        """Calculates a high-quality relative pseudo-depth map using bilateral filtering and intensity gradients."""
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Smooth image to form macro topography
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Calculate morphological gradient to highlight features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_grad = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        
        # Combine intensity (elevation height) with high-frequency structural features
        depth_composite = cv2.addWeighted(blurred, 0.7, morph_grad, 0.3, 0)
        
        # Normalize to [0, 255]
        d_min, d_max = depth_composite.min(), depth_composite.max()
        if d_max > d_min:
            depth_uint8 = ((depth_composite - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            depth_uint8 = depth_composite.astype(np.uint8)
            
        return depth_uint8
