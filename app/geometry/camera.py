import numpy as np
from typing import Tuple, Dict, Any, Optional

class CameraIntrinsics:
    """
    Pinhole Camera Intrinsics Model for 2D-to-3D projection.
    Provides explicit calibration state tagging ("Calibrated" vs "Approximate").
    """
    def __init__(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 512,
        fov_degrees: float = 60.0
    ):
        self.width = image_width
        self.height = image_height
        self.is_calibrated = (fx is not None and fy is not None)
        self.calibration_state = "Calibrated" if self.is_calibrated else "Approximate"

        if fx is not None and fy is not None:
            self.fx = float(fx)
            self.fy = float(fy)
            self.cx = float(cx) if cx is not None else float(image_width / 2.0)
            self.cy = float(cy) if cy is not None else float(image_height / 2.0)
        else:
            fov_rad = np.radians(fov_degrees)
            f_est = float((image_width / 2.0) / np.tan(fov_rad / 2.0))
            self.fx = f_est
            self.fy = f_est
            self.cx = float(image_width / 2.0)
            self.cy = float(image_height / 2.0)

    def project_depth_to_camera_space(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert 2D relative depth map (H, W) into 3D camera-space coordinates (H, W, 3).
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = Z
        """
        h, w = depth_map.shape[:2]
        
        u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        z = depth_map.astype(np.float32)
        x = (u_coords - self.cx) * z / self.fx
        y = (v_coords - self.cy) * z / self.fy

        points_3d = np.stack([x, y, z], axis=-1)
        return points_3d

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "image_width": self.width,
            "image_height": self.height,
            "calibration_state": self.calibration_state,
            "description": "User-supplied calibrated camera intrinsics" if self.is_calibrated else "Approximate pinhole intrinsics (Normalized FOV 60°)"
        }
