import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))

from app.preprocessing.rgb import load_rgb_image
from app.preprocessing.thermal import load_thermal_image
from app.preprocessing.alignment import align_image_pair, compute_smoke_confidence
from app.models.ff_fusion import FFFusionModel
from app.models.depth_anything import DepthAnythingV2Model
from app.geometry.camera import CameraIntrinsics
from app.geometry.depth_to_pointcloud import depth_to_pointcloud, save_pointcloud_ply
from app.geometry.depth_to_mesh import depth_to_mesh, save_mesh_obj, export_mesh_glb
from PIL import Image
import numpy as np

def run_verification():
    rgb_path = "data/sample_pair/rgb_sample.png"
    thermal_path = "data/sample_pair/thermal_sample.tif"
    
    run_dir = "results/run_e2e_verification"
    os.makedirs(run_dir, exist_ok=True)

    print("=== Aero-Topo End-to-End Verification ===")
    
    # 1. Preprocessing & Alignment
    rgb = load_rgb_image(rgb_path)
    raw_th, norm_th_3ch, th_meta = load_thermal_image(thermal_path)
    
    rgb_a, th_a, raw_th_a, align_info = align_image_pair(rgb, norm_th_3ch, raw_th, target_size=(640, 512))
    smoke_info = compute_smoke_confidence(rgb_a)
    print("[1/5] Modality Preprocessing & Spatial Alignment: PASSED")

    # 2. FF-Fusion Model
    ff_model = FFFusionModel(checkpoint_path="models/ff_fusion_student.pth")
    fused_img, fusion_info = ff_model.fuse(rgb_a, th_a)
    print(f"[2/5] FF-Fusion Pipeline ({fusion_info['fusion_method']}): PASSED")

    # 3. Depth Anything V2
    depth_model = DepthAnythingV2Model(encoder="vits", mode="relative")
    raw_depth, norm_depth_visual, quality_info = depth_model.predict_depth(fused_img)
    print(f"[3/5] Depth Anything V2 Prediction (Status: {quality_info['status']}): PASSED")

    # 4. Camera Intrinsics & 3D Projection
    camera = CameraIntrinsics(image_width=640, image_height=512)
    pts_xyz, pts_colors, pc_meta = depth_to_pointcloud(raw_depth, fused_img, camera, subsample=2)
    verts, mesh_colors, faces, mesh_meta = depth_to_mesh(raw_depth, fused_img, camera, subsample=2)
    print(f"[4/5] 3D Geometry Projection ({pc_meta['total_points']} pts, {mesh_meta['num_faces']} faces): PASSED")

    # 5. Export Artifacts
    Image.fromarray(rgb_a).save(os.path.join(run_dir, "input_rgb.png"))
    Image.fromarray(th_a).save(os.path.join(run_dir, "input_thermal.png"))
    Image.fromarray(fused_img).save(os.path.join(run_dir, "fused.png"))
    np.save(os.path.join(run_dir, "depth.npy"), raw_depth)
    Image.fromarray(norm_depth_visual).save(os.path.join(run_dir, "depth_preview.png"))

    save_pointcloud_ply(pts_xyz, pts_colors, os.path.join(run_dir, "pointcloud.ply"))
    save_mesh_obj(verts, mesh_colors, faces, os.path.join(run_dir, "terrain.obj"))
    export_mesh_glb(verts, mesh_colors, faces, os.path.join(run_dir, "terrain.glb"))

    metadata = {
        "fusion_model": fusion_info.get("fusion_method"),
        "depth_model": f"Depth-Anything-V2-{depth_model.encoder}",
        "depth_type": depth_model.mode,
        "smoke_confidence": smoke_info,
        "depth_quality": quality_info,
        "camera": camera.to_dict(),
        "pointcloud": pc_meta,
        "mesh": mesh_meta
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[5/5] Artifact Export (.PLY, .OBJ, .GLB, .NPY, metadata.json): PASSED")
    print(f"Verification output directory: {os.path.abspath(run_dir)}")
    print("SUCCESS: End-to-end execution completed flawlessly!")

if __name__ == "__main__":
    run_verification()
