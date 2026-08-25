import os
import time
import json
import numpy as np
import cv2
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.api.model_manager import ModelManager
from app.preprocessing.rgb import load_rgb_image
from app.preprocessing.thermal import load_thermal_image
from app.preprocessing.alignment import align_image_pair, compute_smoke_confidence
from app.geometry.camera import CameraIntrinsics
from app.geometry.depth_to_pointcloud import depth_to_pointcloud, save_pointcloud_ply
from app.geometry.depth_to_mesh import depth_to_mesh, save_mesh_obj, export_mesh_glb

router = APIRouter(prefix="/api", tags=["Inference"])

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class AnalysisRequest(BaseModel):
    rgb_path: str
    thermal_path: str
    enable_generative: bool = False
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

@router.get("/status")
def get_system_status():
    manager = ModelManager()
    return manager.get_status()

@router.post("/analyze")
def analyze(req: AnalysisRequest):
    """
    Primary Aero-Topo Perception & Relative-Depth Geometry Pipeline:
    RGB + Thermal -> Alignment -> FF-Fusion -> Depth Anything V2 -> 3D Relative-Depth Surface.
    """
    if not os.path.exists(req.rgb_path):
        raise HTTPException(status_code=404, detail=f"RGB file not found: {req.rgb_path}")
    if not os.path.exists(req.thermal_path):
        raise HTTPException(status_code=404, detail=f"Thermal file not found: {req.thermal_path}")

    t_start = time.time()
    run_id = f"run_{int(time.time())}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manager = ModelManager()

    try:
        # 1. Load modalities
        rgb_img = load_rgb_image(req.rgb_path)
        raw_thermal, norm_thermal_3ch, th_meta = load_thermal_image(req.thermal_path)

        # 2. Alignment & Preprocessing
        rgb_aligned, thermal_3ch_aligned, raw_thermal_aligned, align_info = align_image_pair(
            rgb_img, norm_thermal_3ch, raw_thermal, target_size=(640, 512)
        )

        # 3. Qualitative Smoke Confidence Heuristic
        smoke_info = compute_smoke_confidence(rgb_aligned)

        # 4. Primary Model — FF-Fusion
        fused_img, fusion_info = manager.ff_fusion.fuse(rgb_aligned, thermal_3ch_aligned)

        # 5. Primary Depth Model — Depth Anything V2 (Relative Depth)
        raw_depth, norm_depth_visual, depth_quality = manager.depth_anything.predict_depth(fused_img)

        # 6. Camera Intrinsics Model (Explicit Calibrated vs Approximate State)
        camera = CameraIntrinsics(
            fx=req.fx, fy=req.fy, cx=req.cx, cy=req.cy,
            image_width=640, image_height=512
        )

        # 7. 3D Geometry Projection (Pointcloud & Mesh using RAW float32 depth)
        pts_xyz, pts_colors, pc_meta = depth_to_pointcloud(raw_depth, fused_img, camera, subsample=2)
        vertices, mesh_colors, faces, mesh_meta = depth_to_mesh(raw_depth, fused_img, camera, subsample=2)

        # 8. Save Artifacts to run_dir
        input_rgb_path = os.path.join(run_dir, "input_rgb.png")
        input_thermal_path = os.path.join(run_dir, "input_thermal.png")
        fused_path = os.path.join(run_dir, "fused.png")
        depth_npy_path = os.path.join(run_dir, "depth.npy")
        depth_preview_path = os.path.join(run_dir, "depth_preview.png")
        ply_path = os.path.join(run_dir, "pointcloud.ply")
        obj_path = os.path.join(run_dir, "terrain.obj")
        glb_path = os.path.join(run_dir, "terrain.glb")
        meta_json_path = os.path.join(run_dir, "metadata.json")

        Image.fromarray(rgb_aligned).save(input_rgb_path)
        Image.fromarray(thermal_3ch_aligned).save(input_thermal_path)
        Image.fromarray(fused_img).save(fused_path)
        np.save(depth_npy_path, raw_depth)
        Image.fromarray(norm_depth_visual).save(depth_preview_path)

        save_pointcloud_ply(pts_xyz, pts_colors, ply_path)
        save_mesh_obj(vertices, mesh_colors, faces, obj_path)
        try:
            export_mesh_glb(vertices, mesh_colors, faces, glb_path)
            glb_created = True
        except Exception:
            glb_created = False

        # 9. Optional Generative AI Branch (Comparison only)
        generated_rgb_path = None
        generative_info = None
        if req.enable_generative:
            gen_img, generative_info = manager.generative_client.generate_synthetic_rgb(thermal_3ch_aligned)
            if gen_img is not None:
                generated_rgb_path = os.path.join(run_dir, "generated_rgb.png")
                Image.fromarray(gen_img).save(generated_rgb_path)

        # 10. Save Metadata JSON
        processing_time_sec = round(time.time() - t_start, 3)
        metadata = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_sec": processing_time_sec,
            "fusion_model": fusion_info.get("fusion_method", "FF-Fusion"),
            "depth_model": f"Depth-Anything-V2-{manager.depth_anything.encoder}",
            "depth_type": "Relative",
            "alignment": align_info,
            "smoke_confidence": smoke_info,
            "depth_quality": depth_quality,
            "camera_intrinsics": camera.to_dict(),
            "point_cloud": pc_meta,
            "mesh": mesh_meta,
            "generative": generative_info or {"enabled": False}
        }

        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "processing_time_sec": processing_time_sec,
            "artifacts": {
                "input_rgb": f"/results/{run_id}/input_rgb.png",
                "input_thermal": f"/results/{run_id}/input_thermal.png",
                "fused": f"/results/{run_id}/fused.png",
                "depth_preview": f"/results/{run_id}/depth_preview.png",
                "depth_npy": f"/results/{run_id}/depth.npy",
                "pointcloud_ply": f"/results/{run_id}/pointcloud.ply",
                "terrain_obj": f"/results/{run_id}/terrain.obj",
                "terrain_glb": f"/results/{run_id}/terrain.glb" if glb_created else None,
                "generated_rgb": f"/results/{run_id}/generated_rgb.png" if generated_rgb_path else None,
                "metadata_json": f"/results/{run_id}/metadata.json"
            },
            "metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")

@router.post("/stress-test")
def run_wildfire_stress_test(req: AnalysisRequest):
    """
    Wildfire Stress Test Mode under heavy smoke:
    Compares:
    1. RGB alone -> Depth Anything V2 -> 3D Relative Surface
    2. Thermal IR alone
    3. FF-Fusion -> Depth Anything V2 -> 3D Relative Surface
    Goal: Validate whether multimodal fusion preserves scene geometry better than RGB under smoke.
    """
    if not os.path.exists(req.rgb_path) or not os.path.exists(req.thermal_path):
        raise HTTPException(status_code=404, detail="Input RGB or Thermal file not found.")

    manager = ModelManager()
    rgb_img = load_rgb_image(req.rgb_path)
    raw_thermal, norm_thermal_3ch, _ = load_thermal_image(req.thermal_path)

    rgb_a, thermal_3ch_a, _, _ = align_image_pair(rgb_img, norm_thermal_3ch, raw_thermal, target_size=(640, 512))

    # Pipeline 1: RGB alone
    depth_rgb, visual_rgb, q_rgb = manager.depth_anything.predict_depth(rgb_a)

    # Pipeline 2: FF-Fusion (RGB + Thermal)
    fused, fusion_info = manager.ff_fusion.fuse(rgb_a, thermal_3ch_a)
    depth_fused, visual_fused, q_fused = manager.depth_anything.predict_depth(fused)

    return {
        "status": "SUCCESS",
        "smoke_estimate": compute_smoke_confidence(rgb_a),
        "pipeline_rgb_alone": {
            "name": "RGB Alone",
            "depth_quality": q_rgb
        },
        "pipeline_ff_fusion": {
            "name": "RGB + Thermal (FF-Fusion)",
            "fusion_method": fusion_info.get("fusion_method"),
            "depth_quality": q_fused
        }
    }
