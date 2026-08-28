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
from app.preprocessing.thermal import load_thermal_image, preprocess_thermal_for_cgan
from app.preprocessing.alignment import compute_smoke_confidence
from app.geometry.camera import CameraIntrinsics
from app.geometry.depth_to_pointcloud import depth_to_pointcloud, save_pointcloud_ply
from app.geometry.depth_to_mesh import depth_to_mesh, save_mesh_obj, export_mesh_glb

router = APIRouter(prefix="/api", tags=["Inference"])

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class AnalysisRequest(BaseModel):
    thermal_path: str
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
    Thermal IR -> Pretrained Pix2Pix cGAN -> Generated RGB -> FF-Fusion -> Depth Anything V2 -> 3D Relative-Depth Surface.
    """
    if not os.path.exists(req.thermal_path):
        raise HTTPException(status_code=404, detail=f"Thermal file not found: {req.thermal_path}")

    t_start = time.time()
    run_id = f"run_{int(time.time())}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manager = ModelManager()

    try:
        # 1. Load Thermal Modality
        raw_thermal, norm_thermal_3ch, th_meta = load_thermal_image(req.thermal_path)

        # 2. Preprocess Thermal for cGAN: robust 1st-99th percentile norm -> [H, W] in [-1, 1]
        cgan_in_1ch = preprocess_thermal_for_cgan(raw_thermal, target_size=(256, 256))

        # 3. cGAN Inference (Thermal -> Generated RGB)
        generated_rgb, cgan_info = manager.cgan.generate_rgb(cgan_in_1ch, target_size=(640, 512))
        if generated_rgb is None:
            raise HTTPException(
                status_code=503,
                detail=f"cGAN inference unavailable: {cgan_info.get('message', 'Checkpoint missing or invalid.')}"
            )

        # 4. Multimodal FF-Fusion (Generated RGB + Original Thermal)
        fusion_enabled = manager.config.get("models", {}).get("fusion", {}).get("enabled", True)
        
        if fusion_enabled:
            fused_img, fusion_info = manager.ff_fusion.fuse(generated_rgb, norm_thermal_3ch)
            if fused_img is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"FF-Fusion model unavailable: {fusion_info.get('message', 'Weights missing.')}"
                )
        else:
            # Bypass mode: Use Generated RGB directly
            fused_img = generated_rgb
            fusion_info = {
                "fusion_method": "Bypass (Generated RGB directly)",
                "status": "BYPASS"
            }

        # 5. Depth Anything V2 Relative Depth Estimation
        raw_depth, norm_depth_visual, depth_quality = manager.depth_anything.predict_depth(fused_img)

        # 6. Camera Intrinsics
        camera = CameraIntrinsics(
            fx=req.fx, fy=req.fy, cx=req.cx, cy=req.cy,
            image_width=640, image_height=512
        )

        # 7. 3D Geometry Projection (Pointcloud & Mesh)
        pts_xyz, pts_colors, pc_meta = depth_to_pointcloud(raw_depth, fused_img, camera, subsample=2)
        vertices, mesh_colors, faces, mesh_meta = depth_to_mesh(raw_depth, fused_img, camera, subsample=2)

        # 8. Save Artifacts to run_dir
        input_thermal_path = os.path.join(run_dir, "input_thermal.png")
        generated_rgb_path = os.path.join(run_dir, "generated_rgb.png")
        fused_path = os.path.join(run_dir, "fused.png")
        depth_npy_path = os.path.join(run_dir, "depth.npy")
        depth_preview_path = os.path.join(run_dir, "depth_preview.png")
        ply_path = os.path.join(run_dir, "pointcloud.ply")
        obj_path = os.path.join(run_dir, "terrain.obj")
        glb_path = os.path.join(run_dir, "terrain.glb")
        meta_json_path = os.path.join(run_dir, "metadata.json")

        Image.fromarray(norm_thermal_3ch).save(input_thermal_path)
        Image.fromarray(generated_rgb).save(generated_rgb_path)
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

        # 9. Save Metadata JSON
        processing_time_sec = round(time.time() - t_start, 3)
        metadata = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_sec": processing_time_sec,
            "cgan_model": cgan_info.get("model", "Pix2Pix UNetGenerator"),
            "fusion_model": fusion_info.get("fusion_method", "FF-Fusion"),
            "depth_model": f"Depth-Anything-V2-{manager.depth_anything.encoder}",
            "depth_mode": "relative",
            "thermal_metadata": th_meta,
            "depth_quality": depth_quality,
            "camera_intrinsics": camera.to_dict(),
            "point_cloud": pc_meta,
            "mesh": mesh_meta
        }

        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "processing_time_sec": processing_time_sec,
            "artifacts": {
                "input_thermal": f"/results/{run_id}/input_thermal.png",
                "generated_rgb": f"/results/{run_id}/generated_rgb.png",
                "fused": f"/results/{run_id}/fused.png",
                "depth_preview": f"/results/{run_id}/depth_preview.png",
                "depth_npy": f"/results/{run_id}/depth.npy",
                "pointcloud_ply": f"/results/{run_id}/pointcloud.ply",
                "terrain_obj": f"/results/{run_id}/terrain.obj",
                "terrain_glb": f"/results/{run_id}/terrain.glb" if glb_created else None,
                "metadata_json": f"/results/{run_id}/metadata.json"
            },
            "metadata": metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")
