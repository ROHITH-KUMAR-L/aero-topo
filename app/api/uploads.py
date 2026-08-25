import os
import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("")
async def upload_pair(
    rgb_file: UploadFile = File(...),
    thermal_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload paired RGB + Thermal imagery for analysis.
    """
    session_id = f"session_{uuid.uuid4().hex[:10]}_{int(time.time())}"
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    rgb_filename = f"input_rgb_{rgb_file.filename}"
    thermal_filename = f"input_thermal_{thermal_file.filename}"

    rgb_path = os.path.join(session_dir, rgb_filename)
    thermal_path = os.path.join(session_dir, thermal_filename)

    try:
        rgb_content = await rgb_file.read()
        with open(rgb_path, "wb") as f:
            f.write(rgb_content)

        thermal_content = await thermal_file.read()
        with open(thermal_path, "wb") as f:
            f.write(thermal_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

    return {
        "status": "SUCCESS",
        "session_id": session_id,
        "rgb_path": rgb_path,
        "thermal_path": thermal_path,
        "rgb_size_bytes": len(rgb_content),
        "thermal_size_bytes": len(thermal_content)
    }
