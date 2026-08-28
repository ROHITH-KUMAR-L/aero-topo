import os
import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("")
async def upload_thermal(
    thermal_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload Thermal IR imagery for Aero-Topo cGAN + FF-Fusion + 3D Reconstruction pipeline.
    """
    session_id = f"session_{uuid.uuid4().hex[:10]}_{int(time.time())}"
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    safe_filename = os.path.basename(thermal_file.filename or "thermal_input.png")
    thermal_filename = f"input_thermal_{safe_filename}"
    thermal_path = os.path.join(session_dir, thermal_filename)

    try:
        thermal_content = await thermal_file.read()
        with open(thermal_path, "wb") as f:
            f.write(thermal_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded thermal file: {e}")

    return {
        "status": "SUCCESS",
        "session_id": session_id,
        "thermal_path": thermal_path,
        "thermal_size_bytes": len(thermal_content)
    }

