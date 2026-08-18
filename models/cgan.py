import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_PIX2PIX_WEIGHTS = BASE_DIR / "weights" / "pix2pix"

class DownsamplingBlock(nn.Module):
    """Conv-BatchNorm-LeakyReLU downsampling block."""
    def __init__(self, c_in, c_out, use_norm=True):
        super().__init__()
        # No bias needed when a BatchNorm layer follows the convolution
        block = [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            block.append(nn.BatchNorm2d(c_out))
        block.append(nn.LeakyReLU(0.2))
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):
    """ConvTranspose-BatchNorm-(Dropout)-ReLU upsampling block."""
    def __init__(self, c_in, c_out, use_dropout=False):
        super().__init__()
        block = [
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if use_dropout:
            block.append(nn.Dropout(0.5))
        block.append(nn.ReLU())
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UnetEncoder(nn.Module):
    """C64-C128-C256-C512-C512-C512-C512-C512 encoder."""
    def __init__(self, c_in=3, c_out=512):
        super().__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False)
        self.enc2 = DownsamplingBlock(64, 128)
        self.enc3 = DownsamplingBlock(128, 256)
        self.enc4 = DownsamplingBlock(256, 512)
        self.enc5 = DownsamplingBlock(512, 512)
        self.enc6 = DownsamplingBlock(512, 512)
        self.enc7 = DownsamplingBlock(512, 512)
        self.enc8 = DownsamplingBlock(512, c_out)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        # Deepest activation first so the decoder can consume skips in order
        return [x8, x7, x6, x5, x4, x3, x2, x1]


class UnetDecoder(nn.Module):
    """CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 decoder with skip connections."""
    def __init__(self, c_in=512, c_out=64):
        super().__init__()
        self.dec1 = UpsamplingBlock(c_in, 512, use_dropout=True)
        self.dec2 = UpsamplingBlock(1024, 512, use_dropout=True)
        self.dec3 = UpsamplingBlock(1024, 512, use_dropout=True)
        self.dec4 = UpsamplingBlock(1024, 512)
        self.dec5 = UpsamplingBlock(1024, 256)
        self.dec6 = UpsamplingBlock(512, 128)
        self.dec7 = UpsamplingBlock(256, 64)
        self.dec8 = UpsamplingBlock(128, c_out)

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1)
        x10 = torch.cat([x[2], self.dec2(x9)], 1)
        x11 = torch.cat([x[3], self.dec3(x10)], 1)
        x12 = torch.cat([x[4], self.dec4(x11)], 1)
        x13 = torch.cat([x[5], self.dec5(x12)], 1)
        x14 = torch.cat([x[6], self.dec6(x13)], 1)
        x15 = torch.cat([x[7], self.dec7(x14)], 1)
        return self.dec8(x15)


class Pix2PixGenerator(nn.Module):
    """U-Net Generator for Pix2Pix cGAN (Thermal/SAR to RGB translation).

    Module names (encoder.encN / decoder.decN / head) mirror the state_dict keys of
    the pix2pix_gen_*.pth checkpoints in weights/pix2pix so they load strictly.
    Expects 256x256 3-channel input in [-1, 1] and outputs the same range via Tanh.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = UnetEncoder(c_in=in_channels)
        self.decoder = UnetDecoder()
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))

class ThermalToRGBEngine:
    def __init__(self, model_dir=LOCAL_PIX2PIX_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.model = None
        self.is_weights_loaded = False
        self._initialize_model()

    def _resolve_weights_file(self):
        """Picks the highest-epoch pix2pix generator checkpoint, ignoring discriminators."""
        if not self.model_dir.exists():
            return None

        candidates = [
            p for p in self.model_dir.glob("pix2pix_gen_*.pth")
            if p.stem.rsplit("_", 1)[-1].isdigit()
        ]
        if not candidates:
            return None

        return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    def _initialize_model(self):
        """Loads the Pix2Pix generator weights from the local weights directory."""
        self.model = Pix2PixGenerator().to(self.device)
        self.model.eval()

        pth_file = self._resolve_weights_file()

        if pth_file is None:
            print(f"[Pix2PixEngine] No weights found in {self.model_dir}. Operating with structural enhancement mode.")
            return

        try:
            state_dict = torch.load(pth_file, map_location=self.device, weights_only=True)
            # strict=True so architecture/checkpoint mismatches fail loudly instead of
            # silently running the generator with randomly initialized weights.
            self.model.load_state_dict(state_dict, strict=True)
            self.is_weights_loaded = True
            print(f"[Pix2PixEngine] Loaded weights from {pth_file.name} on device: {self.device}")
        except Exception as e:
            print(f"[Pix2PixEngine] Could not load state_dict from {pth_file.name}: {e}. Using structural enhancement fallback.")

    def translate(self, input_image_uint8: np.ndarray) -> np.ndarray:
        """
        Translates a 256x256 Thermal IR image (uint8 RGB or Gray) to structural 3-channel RGB image.
        If weights are loaded, passes through cGAN generator.
        Otherwise, uses high-frequency contrast terrain enhancement to preserve contours.
        """
        if self.is_weights_loaded and self.model is not None:
            try:
                # Normalize to [-1, 1] for Tanh output Generator
                tensor_in = torch.from_numpy(input_image_uint8).permute(2, 0, 1).float() / 127.5 - 1.0
                tensor_in = tensor_in.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output_tensor = self.model(tensor_in)
                    
                # Denormalize [-1, 1] -> [0, 255]
                output_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                output_np = np.clip(output_np, 0, 255).astype(np.uint8)
                return output_np
            except Exception as e:
                print(f"[Pix2PixEngine] Inference error: {e}, falling back to structural enhancement.")

        # Fallback Structural Enhancement Adapter (preserves terrain contours & edges)
        return self._enhance_thermal_to_rgb(input_image_uint8)

    def _enhance_thermal_to_rgb(self, image_np: np.ndarray) -> np.ndarray:
        """High-frequency terrain detail enhancer when generator weights are missing."""
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for thermal contours
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Apply terrain color mapping (COLORMAP_INFERNO or COLORMAP_JET to represent thermal elevation features)
        colored_rgb = cv2.applyColorMap(equalized, cv2.COLORMAP_VIRIDIS)
        colored_rgb = cv2.cvtColor(colored_rgb, cv2.COLOR_BGR2RGB)
        return colored_rgb
