import os
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AeroTopo.DownloadModels")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

WEIGHTS_REGISTRY = {
    "ff_fusion": {
        "filename": "ff_fusion_student.pth",
        "official_url": "https://github.com/FF-Fusion/FF-Fusion",
        "download_url": "https://github.com/FF-Fusion/FF-Fusion/releases/download/v1.0/ff_fusion_student.pth",
        "description": "FF-Fusion Knowledge Distilled Student Model (~1.34 MB)"
    },
    "depth_anything_v2": {
        "filename": "depth_anything_v2_vits.pth",
        "official_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
        "download_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/model.safetensors",
        "description": "Depth Anything V2 Small relative depth model (~24.8M params)"
    }
}

def download_file(url: str, dest_path: str) -> bool:
    try:
        logger.info(f"Downloading {url} to {dest_path}...")
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {dest_path}")
            return True
        else:
            logger.warning(f"Download returned status code {resp.status_code} for {url}")
            return False
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False

def main():
    logger.info("=== Aero-Topo Pretrained Checkpoint Downloader ===")
    
    for key, info in WEIGHTS_REGISTRY.items():
        dest = os.path.join(MODELS_DIR, info["filename"])
        if os.path.exists(dest):
            logger.info(f"[READY] {info['description']} already present at '{dest}'")
            continue

        logger.info(f"[FETCHING] {info['description']}...")
        success = download_file(info["download_url"], dest)
        if not success:
            logger.warning(
                f"[ACCESS_GATE / UNAVAILABLE] Could not automatically download '{info['filename']}'.\n"
                f"Official Source URL: {info['official_url']}\n"
                f"Please manually download the weights file to: {os.path.abspath(dest)}\n"
                f"Note: The system will operate using frequency-domain spatial gradient fusion fallback."
            )

if __name__ == "__main__":
    main()
