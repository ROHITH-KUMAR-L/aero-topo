import os
try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
import cv2
from app.preprocessing.alignment import align_image_pair, compute_smoke_confidence
from app.geometry.camera import CameraIntrinsics
from app.geometry.depth_to_pointcloud import depth_to_pointcloud
from app.geometry.depth_to_mesh import depth_to_mesh
from app.models.depth_anything import DepthAnythingV2Model
from app.models.ff_fusion import FFFusionModel

def test_alignment():
    rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    thermal_3ch = np.zeros((300, 500, 3), dtype=np.uint8)
    raw_thermal = np.zeros((300, 500), dtype=np.float32)

    rgb_a, th_a, raw_a, info = align_image_pair(rgb, thermal_3ch, raw_thermal, target_size=(640, 512))

    assert rgb_a.shape == (512, 640, 3)
    assert th_a.shape == (512, 640, 3)
    assert raw_a.shape == (512, 640)
    assert info["mismatch_detected"] is True

def test_smoke_confidence_qualitative():
    smoke_img = np.full((100, 100, 3), 180, dtype=np.uint8)
    info = compute_smoke_confidence(smoke_img)

    assert info["estimate_type"] == "Heuristic"
    assert info["smoke_level"] in ["Low", "Medium", "High"]
    assert info["visibility_level"] in ["Low", "Medium", "High"]

def test_camera_intrinsics_calibration_state():
    # Approximate camera intrinsics
    cam_approx = CameraIntrinsics(image_width=640, image_height=512)
    assert cam_approx.calibration_state == "Approximate"
    assert cam_approx.to_dict()["calibration_state"] == "Approximate"

    # Calibrated camera intrinsics
    cam_calib = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=256.0, image_width=640, image_height=512)
    assert cam_calib.calibration_state == "Calibrated"
    assert cam_calib.to_dict()["calibration_state"] == "Calibrated"

def test_pointcloud_and_outlier_filtering():
    depth = np.ones((100, 100), dtype=np.float32) * 5.0
    # Add an extreme outlier point
    depth[50, 50] = 500.0
    
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    cam = CameraIntrinsics(image_width=100, image_height=100)

    pts, colors, pc_meta = depth_to_pointcloud(depth, rgb, cam, subsample=1)
    assert pts.shape[1] == 3
    assert pc_meta["outlier_filtered"] is True

def test_ff_fusion_fallback():
    model = FFFusionModel(checkpoint_path="non_existent_weights.pth")
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    th = np.full((100, 100, 3), 128, dtype=np.uint8)

    fused, info = model.fuse(rgb, th)
    assert fused.shape == (100, 100, 3)
    assert info["status"] == "FALLBACK"

def test_depth_anything_quality_statistics():
    model = DepthAnythingV2Model()
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 100
    raw_depth, norm_visual, quality = model.predict_depth(rgb)

    assert raw_depth.shape == (100, 100)
    assert norm_visual.shape == (100, 100, 3)
    assert "min_depth" in quality
    assert "max_depth" in quality
    assert "mean_depth" in quality
    assert "std_depth" in quality
    assert "percentile_range" in quality
    assert quality["depth_mode"] == "Relative"
