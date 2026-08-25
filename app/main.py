import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from app.api.uploads import router as uploads_router
from app.api.inference import router as inference_router
from app.api.results import router as results_router

# Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "aero_topo.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AeroTopo.Main")

app = FastAPI(
    title="Aero-Topo API",
    description="Smoke-resilient UAV RGB+Thermal fusion, depth estimation, and 3D topographical visualization system.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(uploads_router)
app.include_router(inference_router)
app.include_router(results_router)

# Mount Results Directory for direct URL access
os.makedirs("results", exist_ok=True)
app.mount("/results", StaticFiles(directory="results"), name="results")

# Serve Frontend static assets
frontend_dir = os.path.abspath("frontend")
dist_dir = os.path.join(frontend_dir, "dist")

if os.path.exists(os.path.join(dist_dir, "index.html")):
    if os.path.exists(os.path.join(dist_dir, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    @app.get("/")
    async def serve_frontend_dist():
        return FileResponse(os.path.join(dist_dir, "index.html"))
elif os.path.exists(os.path.join(frontend_dir, "index.html")):
    if os.path.exists(os.path.join(frontend_dir, "src")):
        app.mount("/src", StaticFiles(directory=os.path.join(frontend_dir, "src")), name="src")
    @app.get("/")
    async def serve_frontend_root():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "system": "Aero-Topo"}
