import os
import numpy as np
from typing import Tuple, Dict, Any, Optional
from app.geometry.camera import CameraIntrinsics

def depth_to_pointcloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    camera: CameraIntrinsics,
    subsample: int = 1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Project 2D relative depth map into 3D Point Cloud with statistical outlier filtering.
    """
    points_3d = camera.project_depth_to_camera_space(depth_map)
    
    if subsample > 1:
        points_3d = points_3d[::subsample, ::subsample, :]
        colors = rgb_image[::subsample, ::subsample, :]
    else:
        colors = rgb_image

    points_xyz = points_3d.reshape(-1, 3)
    colors_rgb = (colors.reshape(-1, 3).astype(np.float32) / 255.0)

    # 1. Filter non-finite (NaN / Inf) points
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[finite_mask]
    colors_rgb = colors_rgb[finite_mask]

    # 2. Statistical Outlier Filtering (1st - 99th Percentile Z-clipping)
    if points_xyz.shape[0] > 0:
        z_vals = points_xyz[:, 2]
        p1, p99 = np.percentile(z_vals, [1, 99])
        z_mean = np.mean(z_vals)
        z_std = np.std(z_vals)

        # Keep points within 3 std dev or 1st-99th percentile
        if z_std > 1e-6:
            inlier_mask = (z_vals >= p1) & (z_vals <= p99) & (np.abs(z_vals - z_mean) <= 3.5 * z_std)
            points_xyz = points_xyz[inlier_mask]
            colors_rgb = colors_rgb[inlier_mask]

    metadata = {
        "total_points": int(points_xyz.shape[0]),
        "subsample_stride": subsample,
        "outlier_filtered": True,
        "x_min": float(np.min(points_xyz[:, 0])) if points_xyz.shape[0] > 0 else 0.0,
        "x_max": float(np.max(points_xyz[:, 0])) if points_xyz.shape[0] > 0 else 0.0,
        "y_min": float(np.min(points_xyz[:, 1])) if points_xyz.shape[0] > 0 else 0.0,
        "y_max": float(np.max(points_xyz[:, 1])) if points_xyz.shape[0] > 0 else 0.0,
        "z_min": float(np.min(points_xyz[:, 2])) if points_xyz.shape[0] > 0 else 0.0,
        "z_max": float(np.max(points_xyz[:, 2])) if points_xyz.shape[0] > 0 else 0.0,
    }

    return points_xyz, colors_rgb, metadata

def save_pointcloud_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray, file_path: str):
    """
    Save 3D point cloud to ASCII PLY file format.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    num_points = points_xyz.shape[0]
    colors_uint8 = (colors_rgb * 255.0).clip(0, 255).astype(np.uint8)

    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            x, y, z = points_xyz[i]
            r, g, b = colors_uint8[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
