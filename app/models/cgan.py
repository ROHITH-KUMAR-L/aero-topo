import os
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("AeroTopo.cGAN")

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize
            )
        ]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        self.d1 = DownBlock(in_channels, 64, normalize=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)
        self.d7 = DownBlock(512, 512)
        self.d8 = DownBlock(512, 512, normalize=False)

        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 512, dropout=True)
        self.u4 = UpBlock(1024, 512)
        self.u5 = UpBlock(1024, 256)
        self.u6 = UpBlock(512, 128)
        self.u7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))


class CGANModel:
    """
    Production cGAN Thermal-to-RGB Inference Wrapper.
    Uses pretrained Pix2Pix UNetGenerator (in_channels=1, out_channels=3).
    """
    def __init__(self, checkpoint_path: str = "models/checkpoints/generator_best.pth", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model: Optional[UNetGenerator] = None
        self.is_ready = False
        self.status_message = ""
        self.metadata: Dict[str, Any] = {
            "architecture": "Pix2Pix U-Net Generator",
            "input_channels": 1,
            "output_channels": 3,
            "image_size": 256
        }

        self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(self.checkpoint_path):
            self.status_message = (
                f"cGAN checkpoint not found at '{self.checkpoint_path}'. "
                "Place generator_best.pth in models/checkpoints/ or configure its Hugging Face repository."
            )
            logger.warning(self.status_message)
            self.is_ready = False
            return

        try:
            model = UNetGenerator(in_channels=1, out_channels=3)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

            # Detect state_dict key structure safely
            state_dict = None
            if isinstance(ckpt, dict):
                for key in ["generator_state_dict", "state_dict", "model_state_dict"]:
                    if key in ckpt:
                        state_dict = ckpt[key]
                        break
                if state_dict is None and ("d1.block.0.weight" in ckpt or any(k.startswith("final") for k in ckpt.keys())):
                    state_dict = ckpt

                # Extract metadata if available
                if "epoch" in ckpt: self.metadata["epoch"] = ckpt["epoch"]
                if "val_l1" in ckpt: self.metadata["val_l1"] = ckpt["val_l1"]
                if "dataset" in ckpt: self.metadata["dataset"] = ckpt["dataset"]
            else:
                state_dict = ckpt

            if state_dict is None:
                raise ValueError("Could not extract valid state_dict from checkpoint file.")

            # Clean module. and generator. key prefixes if present
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("module."):
                    new_key = new_key[7:]
                if new_key.startswith("generator."):
                    new_key = new_key[10:]
                clean_state_dict[new_key] = v

            model.load_state_dict(clean_state_dict)
            model.to(self.device)
            model.eval()

            # Execute validation pass with dummy tensor [1, 1, 256, 256]
            with torch.inference_mode():
                dummy_in = torch.zeros(1, 1, 256, 256, device=self.device)
                dummy_out = model(dummy_in)
                if dummy_out.shape != (1, 3, 256, 256):
                    raise ValueError(f"cGAN validation test failed. Expected shape (1, 3, 256, 256), got {dummy_out.shape}")

            self.model = model
            self.is_ready = True
            self.status_message = f"cGAN (Pix2Pix U-Net) loaded successfully on {self.device}."
            logger.info(self.status_message)

        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to load cGAN checkpoint from '{self.checkpoint_path}': {e}"
            logger.error(self.status_message)

    def generate_rgb(self, thermal_1ch: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Runs cGAN thermal-to-RGB translation.
        Input:
            thermal_1ch: 
                - If uint8 or uint16: assumed raw image, will be normalized to [-1, 1].
                - If float32: assumed already preprocessed to [-1, 1].
            target_size: optional (width, height) to resize generated RGB output back to
        Returns:
            generated_rgb: uint8 RGB numpy array (H, W, 3) with values in [0, 255]
            info: metadata dictionary
        """
        if not self.is_ready or self.model is None:
            return None, {
                "status": "UNAVAILABLE",
                "message": self.status_message,
                "label": "cGAN Model Unavailable"
            }

        orig_h, orig_w = thermal_1ch.shape[:2]
        if target_size is None:
            target_size = (orig_w, orig_h)

        try:
            # 1. Normalize thermal input to float32 in [-1, 1]
            th = thermal_1ch.squeeze()
            if th.dtype == np.uint8:
                th_norm = (th.astype(np.float32) / 127.5) - 1.0
            elif th.dtype == np.uint16:
                th_norm = (th.astype(np.float32) / 32767.5) - 1.0
            elif th.dtype == np.float32 and (th.min() < -1.0 or th.max() > 1.0):
                th_min = th.min()
                th_max = th.max()
                if (th_max - th_min) > 1e-6:
                    th_01 = (th - th_min) / (th_max - th_min)
                    th_norm = (th_01 * 2.0) - 1.0
                else:
                    th_norm = np.zeros_like(th)
            else:
                # Assume already [-1, 1]
                th_norm = th

            # 2. Resize to 256x256 for model input
            th_256 = cv2.resize(th_norm, (256, 256), interpolation=cv2.INTER_AREA)

            # 3. Create tensor [1, 1, 256, 256]
            tensor_in = torch.from_numpy(th_256).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # 4. Inference
            with torch.inference_mode():
                tensor_out = self.model(tensor_in)  # Output in range [-1, 1] (Tanh)

            # 5. Convert Tanh output [-1, 1] -> [0, 1] -> uint8 [0, 255]
            out_np = tensor_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_01 = np.clip((out_np + 1.0) / 2.0, 0.0, 1.0)
            out_uint8 = (out_01 * 255.0).astype(np.uint8)

            # 6. Resize back to target_size if required
            out_final = cv2.resize(out_uint8, target_size, interpolation=cv2.INTER_CUBIC)

            return out_final, {
                "status": "SUCCESS",
                "label": "Generated RGB",
                "description": "Thermal-to-Visible Translation via Pix2Pix cGAN",
                "device": str(self.device),
                "model": "UNetGenerator"
            }

        except Exception as e:
            logger.error(f"cGAN inference error: {e}")
            return None, {
                "status": "ERROR",
                "message": str(e),
                "label": "cGAN Inference Error"
            }
