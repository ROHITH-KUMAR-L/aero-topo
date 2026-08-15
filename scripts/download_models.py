import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Define project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

PIX2PIX_MODEL_ID = "yuulind/pix2pix-sar2rgb"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

def download_models():
    """Download pre-trained models from Hugging Face Hub to local directory."""
    print("=" * 60)
    print(" Aero-Topo Model Weight Downloader ")
    print("=" * 60)

    # 1. Download Pix2Pix SAR2RGB / Thermal2RGB weights
    pix2pix_dir = WEIGHTS_DIR / "pix2pix"
    print(f"\n[1/2] Downloading Pix2Pix model ({PIX2PIX_MODEL_ID}) to {pix2pix_dir}...")
    pix2pix_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=PIX2PIX_MODEL_ID,
            local_dir=str(pix2pix_dir),
            local_dir_use_symlinks=False
        )
        print(" -> Pix2Pix weights downloaded successfully!")
    except Exception as e:
        print(f" -> Error downloading Pix2Pix weights: {e}")
        print(" -> Note: Aero-Topo will use its built-in enhancement adapter fallback if weights are absent.")

    # 2. Download Depth Anything V2 weights
    depth_dir = WEIGHTS_DIR / "depth_anything_v2"
    print(f"\n[2/2] Downloading Depth Anything V2 model ({DEPTH_MODEL_ID}) to {depth_dir}...")
    depth_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=DEPTH_MODEL_ID,
            local_dir=str(depth_dir),
            local_dir_use_symlinks=False
        )
        print(" -> Depth Anything V2 weights downloaded successfully!")
    except Exception as e:
        print(f" -> Error downloading Depth Anything V2 weights: {e}")
        print(" -> Note: Aero-Topo will use Hugging Face hub auto-loading or mock depth estimation fallback.")

    print("\nDownload process completed!")

if __name__ == "__main__":
    download_models()
