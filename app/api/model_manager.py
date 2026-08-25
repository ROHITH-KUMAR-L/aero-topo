import os
import yaml
import logging
from typing import Optional, Dict, Any
from app.models.ff_fusion import FFFusionModel
from app.models.depth_anything import DepthAnythingV2Model
from app.models.robofirefusenet import RoboFireFuseNetModel
from app.models.generative_api import GenerativeRGBClient

logger = logging.getLogger("AeroTopo.ModelManager")

class ModelManager:
    """
    Singleton Lazy Model Manager.
    Ensures FF-Fusion, Depth Anything V2, RoboFireFuseNet, and Generative API clients
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
        
        self._ff_fusion: Optional[FFFusionModel] = None
        self._depth_anything: Optional[DepthAnythingV2Model] = None
        self._robofirefusenet: Optional[RoboFireFuseNetModel] = None
        self._generative_client: Optional[GenerativeRGBClient] = None
        self._initialized = True

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    @property
    def ff_fusion(self) -> FFFusionModel:
        if self._ff_fusion is None:
            cfg = self.config.get("models", {}).get("fusion", {})
            self._ff_fusion = FFFusionModel(
                checkpoint_path=cfg.get("local_path", "models/ff_fusion_student.pth"),
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
                checkpoint_path=cfg.get("local_path", "models/depth_anything_v2_vits.pth"),
                device=cfg.get("device", "cuda"),
                fp16=cfg.get("fp16", True)
            )
        return self._depth_anything

    @property
    def robofirefusenet(self) -> RoboFireFuseNetModel:
        if self._robofirefusenet is None:
            cfg = self.config.get("models", {}).get("robofirefusenet", {})
            self._robofirefusenet = RoboFireFuseNetModel(
                checkpoint_path=cfg.get("local_path", "models/robofirefusenet.pth"),
                enabled=cfg.get("enabled", False)
            )
        return self._robofirefusenet

    @property
    def generative_client(self) -> GenerativeRGBClient:
        if self._generative_client is None:
            cfg = self.config.get("generative", {})
            self._generative_client = GenerativeRGBClient(
                enabled=cfg.get("enabled", False),
                model=cfg.get("model", "gpt-image-2"),
                prompt=cfg.get("prompt", "")
            )
        return self._generative_client

    def get_status(self) -> Dict[str, Any]:
        return {
            "ff_fusion": {
                "ready": self.ff_fusion.is_ready,
                "status_message": self.ff_fusion.status_message
            },
            "depth_anything": {
                "ready": self.depth_anything.is_ready,
                "encoder": self.depth_anything.encoder,
                "mode": self.depth_anything.mode,
                "status_message": self.depth_anything.status_message
            },
            "robofirefusenet": {
                "enabled": self.robofirefusenet.enabled,
                "ready": self.robofirefusenet.is_ready
            },
            "generative_api": {
                "enabled": self.generative_client.enabled,
                "ready": self.generative_client.is_available()[0],
                "status_message": self.generative_client.is_available()[1]
            }
        }
