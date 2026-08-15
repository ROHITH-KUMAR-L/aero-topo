import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import AeroTopoPipeline
from utils.preprocessing import read_image_bytes

app = FastAPI(
    title="Aero-Topo API",
    description="Thermal-to-3D Topographical Reconstruction Engine using Pix2Pix cGAN, Depth Anything V2, and Three.js WebGL rendering.",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline_instance = None

@app.on_event("startup")
def startup_event():
    global pipeline_instance
    print("[FastAPI] Initializing Aero-Topo AI Pipeline...")
    pipeline_instance = AeroTopoPipeline()
    print("[FastAPI] Server startup complete!")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "pipeline_ready": pipeline_instance is not None,
        "device": str(pipeline_instance.cgan_engine.device) if pipeline_instance else "unknown",
        "pix2pix_weights_loaded": pipeline_instance.cgan_engine.is_weights_loaded if pipeline_instance else False,
        "depth_weights_loaded": pipeline_instance.depth_engine.is_weights_loaded if pipeline_instance else False
    }

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    use_failsafe: bool = Form(False)
):
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        raw_img_np = read_image_bytes(image_bytes, filename=file.filename)
        
        result = pipeline_instance.process_thermal_image(
            raw_thermal_np=raw_img_np,
            use_failsafe=use_failsafe
        )
        return result
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static directory for Three.js WebGL frontend
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h2>Aero-Topo API Running. Static frontend index.html not found.</h2>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
