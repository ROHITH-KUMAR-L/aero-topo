import time
import numpy as np
from models.cgan import ThermalToRGBEngine
from models.depth_engine import DepthEstimationEngine
from pipeline.failsafe import run_canny_failsafe
from utils.preprocessing import resize_and_normalize, numpy_to_base64_data_url

class AeroTopoPipeline:
    def __init__(self):
        print("[AeroTopoPipeline] Initializing pipeline nodes...")
        self.cgan_engine = ThermalToRGBEngine()
        self.depth_engine = DepthEstimationEngine()
        print("[AeroTopoPipeline] All nodes ready!")

    def process_thermal_image(
        self,
        raw_thermal_np: np.ndarray,
        use_failsafe: bool = False,
        target_size: tuple = (256, 256)
    ) -> dict:
        """
        Runs the full 3-stage sequential pipeline on input thermal image:
        Input Thermal IR -> Stage 1: Pix2Pix cGAN (Thermal to RGB) -> Stage 2: Depth Anything V2 -> Output Z-map.
        """
        start_time = time.time()
        
        # 1. Resize and prepare thermal input tensor
        thermal_resized = resize_and_normalize(raw_thermal_np, target_size=target_size)
        
        # 2. Stage 1: Thermal to RGB Translation (or Failsafe edge extraction)
        if use_failsafe:
            print("[Pipeline] Running Emergency Failsafe (Canny Edge Extraction)...")
            rgb_feature = run_canny_failsafe(thermal_resized)
            stage1_mode = "OpenCV Canny Edge Failsafe"
        else:
            print("[Pipeline] Running Stage 1 Generative Translation (Pix2Pix)...")
            rgb_feature = self.cgan_engine.translate(thermal_resized)
            stage1_mode = "Pix2Pix SAR/Thermal-to-RGB (yuulind/pix2pix-sar2rgb)" if self.cgan_engine.is_weights_loaded else "Enhanced Structural Thermal-to-RGB"

        # 3. Stage 2: Monocular Depth Estimation
        print("[Pipeline] Running Stage 2 Depth Estimation (Depth Anything V2)...")
        depth_zmap = self.depth_engine.estimate_depth(rgb_feature)
        stage2_mode = "Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf)" if self.depth_engine.is_weights_loaded else "Gradient Elevation Map Fallback"

        elapsed_sec = round(time.time() - start_time, 3)

        # 4. Generate Base64 Data URLs for frontend rendering
        thermal_b64 = numpy_to_base64_data_url(thermal_resized)
        rgb_b64 = numpy_to_base64_data_url(rgb_feature)
        depth_b64 = numpy_to_base64_data_url(depth_zmap, is_grayscale=True)

        return {
            "status": "success",
            "execution_time_sec": elapsed_sec,
            "stage1_mode": stage1_mode,
            "stage2_mode": stage2_mode,
            "use_failsafe": use_failsafe,
            "dimensions": {"width": target_size[0], "height": target_size[1]},
            "images": {
                "thermal_input": thermal_b64,
                "rgb_output": rgb_b64,
                "depth_map": depth_b64
            }
        }
