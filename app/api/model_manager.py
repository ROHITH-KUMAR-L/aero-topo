import os
import yaml
import logging
from typing import Optional, Dict, Any
from app.models.cgan import CGANModel
from app.models.ff_fusion import FFFusionModel
from app.models.depth_anything import DepthAnythingV2Model

logger = logging.getLogger("AeroTopo.ModelManager")

class ModelManager:
    """
    Singleton Lazy Model Manager.
    Ensures cGAN, FF-Fusion, and Depth Anything V2
    are loaded once into GPU/CPU memory and cached across inference requests.
    """
    _instance: Optional['ModelManager'] = None

    def __new__(cls, config_path: str = "app/config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "app/config/config.yaml"):
        if getattr(self, '_initialized', False):
            return

        self.config_path = config_path
        self.config = self._load_config()
        
        self._cgan: Optional[CGANModel] = None
        self._ff_fusion: Optional[FFFusionModel] = None
        self._depth_anything: Optional[DepthAnythingV2Model] = None
        self._initialized = True

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def cgan(self) -> CGANModel:
        if self._cgan is None:
            cfg = self.config.get("models", {}).get("cgan", {})
            self._cgan = CGANModel(
                checkpoint_path=cfg.get("checkpoint_path", "models/checkpoints/generator_best.pth"),
                device=cfg.get("device", "cuda")
            )
        return self._cgan

    @property
    def ff_fusion(self) -> Optional[FFFusionModel]:
        cfg = self.config.get("models", {}).get("fusion", {})
        if not cfg.get("enabled", True):
            return None

        if self._ff_fusion is None:
            self._ff_fusion = FFFusionModel(
                checkpoint_path=cfg.get("checkpoint_path", "models/checkpoints/ff_fusion.pth"),
                device=cfg.get("device", "cuda"),
                fp16=cfg.get("fp16", True)
            )
        return self._ff_fusion

    @property
    def depth_anything(self) -> DepthAnythingV2Model:
        if self._depth_anything is None:
            cfg = self.config.get("models", {}).get("depth", {})
            self._depth_anything = DepthAnythingV2Model(
                encoder=cfg.get("encoder", "vits"),
                mode=cfg.get("mode", "relative"),
                checkpoint_path=cfg.get("checkpoint_path", "models/checkpoints/depth_anything_v2.pth"),
                device=cfg.get("device", "cuda"),
                fp16=cfg.get("fp16", True)
            )
        return self._depth_anything

    def get_status(self) -> Dict[str, Any]:
        fusion_cfg = self.config.get("models", {}).get("fusion", {})
        fusion_enabled = fusion_cfg.get("enabled", True)
        
        if fusion_enabled:
            fusion_status = {
                "enabled": True,
                "available": self.ff_fusion.is_ready if self.ff_fusion else False,
                "mode": "active",
                "checkpoint": os.path.basename(self.ff_fusion.checkpoint_path) if self.ff_fusion else "None",
                "status_message": self.ff_fusion.status_message if self.ff_fusion else "Not initialized"
            }
        else:
            fusion_status = {
                "enabled": False,
                "available": False,
                "mode": "bypass",
                "status_message": "Fusion bypassed via configuration."
            }

        return {
            "cgan": {
                "available": self.cgan.is_ready,
                "checkpoint": os.path.basename(self.cgan.checkpoint_path),
                "architecture": self.cgan.metadata.get("architecture", "Pix2Pix U-Net Generator"),
                "input_channels": self.cgan.metadata.get("input_channels", 1),
                "output_channels": self.cgan.metadata.get("output_channels", 3),
                "image_size": self.cgan.metadata.get("image_size", 256),
                "status_message": self.cgan.status_message
            },
            "ff_fusion": fusion_status,
            "depth_anything_v2": {
                "available": self.depth_anything.is_ready,
                "variant": self.depth_anything.encoder,
                "depth_mode": self.depth_anything.mode,
                "checkpoint": os.path.basename(self.depth_anything.checkpoint_path),
                "status_message": self.depth_anything.status_message
            }
        }

