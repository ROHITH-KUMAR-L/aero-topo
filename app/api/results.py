import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any

router = APIRouter(prefix="/results", tags=["Results"])

RESULTS_DIR = "results"

@router.get("/{run_id}/{filename}")
def get_result_file(run_id: str, filename: str):
    """
    Retrieve run artifact file (PNG, NPY, PLY, OBJ, GLB, JSON).
    """
    file_path = os.path.join(RESULTS_DIR, run_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename} for run {run_id}")

    media_type = "application/octet-stream"
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".json"):
        media_type = "application/json"
    elif filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".obj"):
        media_type = "text/plain"
    elif filename.endswith(".ply"):
        media_type = "text/plain"
    elif filename.endswith(".npy"):
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type)
