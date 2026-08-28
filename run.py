import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AeroTopo.Run")

def check_python_environment():
    logger.info(f"Python Version: {sys.version.split()[0]}")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        logger.info(f"PyTorch Version: {torch.__version__} | CUDA Available: {cuda_avail}")
        if cuda_avail:
            logger.info(f"Target GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch is not installed. Please install requirements from requirements.txt")

def download_weights():
    logger.info("Verifying model weights...")
    dl_script = os.path.join("scripts", "download_models.py")
    if os.path.exists(dl_script):
        try:
            subprocess.run([sys.executable, dl_script], check=True)
        except Exception as e:
            logger.warning(f"Weights check script returned warning: {e}")

def build_frontend_if_needed():
    frontend_dir = os.path.abspath("frontend")
    if not os.path.exists(frontend_dir):
        return

    dist_index = os.path.join(frontend_dir, "dist", "index.html")
    root_index = os.path.join(frontend_dir, "index.html")

    if not os.path.exists(dist_index) and os.path.exists(root_index):
        logger.info("Attempting frontend static build (Vite)...")
        try:
            # Run npm build if node is installed
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, shell=True, check=False)
        except Exception:
            logger.info("Node/Nite build skipped. FastAPI will serve root frontend files directly.")

def start_server(host="127.0.0.1", port=8000):
    logger.info("==================================================")
    logger.info("STARTING AERO-TOPO SYSTEM SERVER")
    logger.info("==================================================")
    logger.info(f"Backend API:  http://{host}:{port}")
    logger.info(f"Frontend UI:  http://{host}:{port}")
    logger.info("==================================================")

    try:
        import uvicorn
        uvicorn.run("app.main:app", host=host, port=port, reload=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

def verify_models():
    logger.info("Initializing model checkpoints...")
    try:
        from app.api.model_manager import ModelManager
        manager = ModelManager()
        status = manager.get_status()
        
        logger.info("Model Status:")
        for name, info in status.items():
            avail = info.get("available", False)
            ckpt = info.get("checkpoint", "unknown")
            msg = info.get("status_message", "")
            state_str = "[READY]" if avail else "[UNAVAILABLE]"
            logger.info(f"  {state_str} {name.upper()}: checkpoint='{ckpt}' | {msg}")
    except Exception as e:
        logger.warning(f"Model status check warning: {e}")

def main():
    check_python_environment()
    download_weights()
    verify_models()
    build_frontend_if_needed()
    start_server()


if __name__ == "__main__":
    main()
