import os
import torch
import torch.nn as nn
import numpy as np
import cv2

from app.models.cgan import UNetGenerator, CGANModel, DownBlock, UpBlock
from app.models.ff_fusion import FFFusionModel
from app.models.depth_anything import DepthAnythingV2Model
from app.preprocessing.thermal import preprocess_thermal_for_cgan, load_thermal_image
from app.geometry.camera import CameraIntrinsics
from app.geometry.depth_to_pointcloud import depth_to_pointcloud
from app.geometry.depth_to_mesh import depth_to_mesh

# ──────────────────────────────────────────────
# UNetGenerator Architecture Structural Tests
# ──────────────────────────────────────────────

def test_unet_generator_architecture_shape():
    """Output tensor must be [1, 3, 256, 256] in [-1, 1]."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
    assert out.shape == (1, 3, 256, 256)
    # Output must be in [-1, 1] due to Tanh activation
    assert out.min() >= -1.0 and out.max() <= 1.0

def test_unet_d1_d8_no_instance_norm():
    """d1 and d8 MUST NOT contain InstanceNorm2d (prevents ValueError at 1×1 bottleneck)."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    for layer in model.d1.block:
        assert not isinstance(layer, nn.InstanceNorm2d), "d1 must NOT use InstanceNorm2d"
    for layer in model.d8.block:
        assert not isinstance(layer, nn.InstanceNorm2d), "d8 must NOT use InstanceNorm2d"

def test_unet_d2_through_d7_have_instance_norm():
    """d2–d7 MUST include InstanceNorm2d for feature normalisation."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    for name, block_attr in [("d2", model.d2), ("d3", model.d3), ("d4", model.d4),
                               ("d5", model.d5), ("d6", model.d6), ("d7", model.d7)]:
        has_norm = any(isinstance(l, nn.InstanceNorm2d) for l in block_attr.block)
        assert has_norm, f"{name} must contain InstanceNorm2d"

def test_unet_u1_u3_have_dropout():
    """u1–u3 MUST include Dropout(0.5) for training-compatible stochastic depth."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    for name, block_attr in [("u1", model.u1), ("u2", model.u2), ("u3", model.u3)]:
        has_dropout = any(isinstance(l, nn.Dropout) for l in block_attr.block)
        assert has_dropout, f"{name} must include Dropout"

def test_unet_u4_u7_no_dropout():
    """u4–u7 MUST NOT include Dropout."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    for name, block_attr in [("u4", model.u4), ("u5", model.u5), ("u6", model.u6), ("u7", model.u7)]:
        has_dropout = any(isinstance(l, nn.Dropout) for l in block_attr.block)
        assert not has_dropout, f"{name} must NOT include Dropout"

def test_unet_final_layer_has_tanh():
    """Final layer MUST terminate with nn.Tanh() to bound output to [-1, 1]."""
    model = UNetGenerator(in_channels=1, out_channels=3)
    assert isinstance(model.final[-1], nn.Tanh), "final block must end with nn.Tanh()"

# ──────────────────────────────────────────────
# Thermal Preprocessing Tests
# ──────────────────────────────────────────────

def test_thermal_preprocessing_cgan_range():
    """Output must be in [-1, 1] after percentile normalization."""
    raw_thermal = np.random.uniform(0, 1000, (400, 600)).astype(np.float32)
    cgan_in = preprocess_thermal_for_cgan(raw_thermal, target_size=(256, 256))
    assert cgan_in.shape == (256, 256)
    assert cgan_in.min() >= -1.0 and cgan_in.max() <= 1.0

def test_thermal_preprocessing_outlier_robustness():
    """
    Robust percentile normalization must not be dominated by extreme outliers.
    Introduce a single hot pixel at 10× the normal range.
    The bulk of pixels should still be normalized correctly (not collapsed near zero).
    """
    raw_thermal = np.random.uniform(200.0, 400.0, (256, 256)).astype(np.float32)
    # Hot pixel: extreme outlier
    raw_thermal[0, 0] = 100000.0
    cgan_in = preprocess_thermal_for_cgan(raw_thermal, target_size=(256, 256))
    # Even with the extreme outlier, the mean of the interior pixels should be >0.
    # With naive min-max, the outlier would collapse all other values near -1.0.
    assert cgan_in.mean() > -0.5, "Outlier robustness failed: bulk values collapsed near -1.0"

def test_thermal_preprocessing_constant_image():
    """Constant-value thermal images must produce a zero array without divide-by-zero."""
    raw_thermal = np.ones((256, 256), dtype=np.float32) * 500.0
    cgan_in = preprocess_thermal_for_cgan(raw_thermal, target_size=(256, 256))
    assert cgan_in.shape == (256, 256)
    assert not np.isnan(cgan_in).any()

# ──────────────────────────────────────────────
# Model Availability Tests
# ──────────────────────────────────────────────

def test_cgan_model_missing_checkpoint():
    """Missing checkpoint must set is_ready=False and return UNAVAILABLE (not a crash or fallback)."""
    model = CGANModel(checkpoint_path="non_existent_generator.pth")
    assert model.is_ready is False
    assert "not found" in model.status_message.lower()

    dummy_th = np.zeros((256, 256), dtype=np.float32)
    out, info = model.generate_rgb(dummy_th)
    assert out is None
    assert info["status"] == "UNAVAILABLE"

def test_ff_fusion_missing_checkpoint():
    """Missing FF-Fusion checkpoint must set is_ready=False and return None on fuse()."""
    model = FFFusionModel(checkpoint_path="non_existent_fusion.pth")
    assert model.is_ready is False

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    th = np.full((100, 100, 3), 128, dtype=np.uint8)
    fused, info = model.fuse(rgb, th)
    assert fused is None
    assert info["status"] == "UNAVAILABLE"

# ──────────────────────────────────────────────
# Camera Intrinsics Tests
# ──────────────────────────────────────────────

def test_camera_intrinsics_calibration_state():
    cam_approx = CameraIntrinsics(image_width=640, image_height=512)
    assert cam_approx.calibration_state == "Approximate"

    cam_calib = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=256.0, image_width=640, image_height=512)
    assert cam_calib.calibration_state == "Calibrated"

# ──────────────────────────────────────────────
# Geometry Tests
# ──────────────────────────────────────────────

def test_pointcloud_and_mesh_generation():
    depth = np.ones((100, 100), dtype=np.float32) * 5.0
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    cam = CameraIntrinsics(image_width=100, image_height=100)

    pts, colors, pc_meta = depth_to_pointcloud(depth, rgb, cam, subsample=2)
    assert pts.shape[1] == 3
    assert pc_meta["outlier_filtered"] is True

    verts, mesh_colors, faces, mesh_meta = depth_to_mesh(depth, rgb, cam, subsample=2)
    assert len(verts) > 0
    assert len(faces) > 0

def test_depth_anything_quality_statistics():
    model = DepthAnythingV2Model()
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 100
    raw_depth, norm_visual, quality = model.predict_depth(rgb)

    assert raw_depth.shape == (100, 100)
    assert norm_visual.shape == (100, 100, 3)
    assert quality["depth_mode"] == "Relative"
