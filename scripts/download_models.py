import os
import sys
import shutil
import logging
from pathlib import Path

import yaml
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "app" / "config" / "config.yaml"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AeroTopo.DownloadModels")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def download_from_huggingface(repo_id, filename, dest_path, token=""):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        kwargs = {"repo_id": repo_id, "filename": filename}
        if token:
            kwargs["token"] = token

        logger.info(f"Downloading {repo_id}/{filename}")
        downloaded = Path(hf_hub_download(**kwargs))

        if downloaded.exists():
            shutil.copy2(downloaded, dest_path)
            logger.info(f"[SUCCESS] Saved to {dest_path}")
            return True

    except Exception as e:
        logger.warning(f"hf_hub_download failed: {e}")

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        logger.info(f"Trying direct download: {url}")
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        with dest_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        logger.info(f"[SUCCESS] Saved to {dest_path}")
        return True

    except Exception as e:
        logger.error(f"[FAILED] {e}")
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return False


def process_model(name, cfg, default_filename, description, hf_token):
    logger.info(f"Checking [{name}] {description}")

    path = Path(
        cfg.get(
            "checkpoint_path",
            str(CHECKPOINTS_DIR / default_filename)
        )
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"[FOUND] {path} ({size_mb:.2f} MB)")
        return True

    logger.info(f"[MISSING] {path}")

    hf_cfg = cfg.get("huggingface", {}) or {}
    repo_id = str(hf_cfg.get("repo_id", "")).strip()
    filename = str(
        hf_cfg.get("filename", default_filename)
    ).strip()

    if (
        not repo_id
        or repo_id.startswith("YOUR_")
        or "REPLACE" in repo_id
    ):
        logger.warning(f"[UNCONFIGURED] No valid HF repo for {name}")
        logger.warning(f"Place the checkpoint manually at: {path}")
        return False

    return download_from_huggingface(
        repo_id,
        filename,
        path,
        hf_token
    )


def main():
    logger.info("=" * 60)
    logger.info("AERO-TOPO MODEL CHECKPOINT DOWNLOADER")
    logger.info("=" * 60)

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        sys.exit(1)

    models = config.get("models", {}) or {}
    hf_token = os.getenv("HF_TOKEN", "").strip()

    process_model(
        "cgan",
        models.get("cgan", {}),
        "generator_best.pth",
        "Pix2Pix cGAN Generator",
        hf_token
    )

    process_model(
        "fusion",
        models.get("fusion", {}),
        "ff_fusion.pth",
        "FF-Fusion",
        hf_token
    )

    depth_cfg = models.get("depth", {}) or {}
    depth_path = Path(
        depth_cfg.get(
            "checkpoint_path",
            str(CHECKPOINTS_DIR / "depth_anything_v2.pth")
        )
    )

    if not depth_path.is_absolute():
        depth_path = PROJECT_ROOT / depth_path

    if depth_path.exists():
        size_mb = depth_path.stat().st_size / (1024 * 1024)
        logger.info(f"[FOUND] Depth Anything V2: {depth_path} ({size_mb:.2f} MB)")
    else:
        logger.info("[INFO] Depth Anything V2 local checkpoint not found.")
        logger.info("[INFO] Runtime uses the Transformers/Hugging Face loader.")

    logger.info("=" * 60)
    logger.info("MODEL CHECKPOINT VERIFICATION COMPLETE")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Checkpoints: {CHECKPOINTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()