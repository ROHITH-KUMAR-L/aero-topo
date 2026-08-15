import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_PIX2PIX_WEIGHTS = BASE_DIR / "weights" / "pix2pix"

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False) if down else \
               nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        layers = [conv, nn.BatchNorm2d(out_channels)]
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Pix2PixGenerator(nn.Module):
    """Standard U-Net Generator for Pix2Pix cGAN (Thermal/SAR to RGB translation)."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(in_channels, 64, 4, 2, 1) # 128x128
        self.e2 = UNetBlock(64, 128, down=True, act="leaky") # 64x64
        self.e3 = UNetBlock(128, 256, down=True, act="leaky") # 32x32
        self.e4 = UNetBlock(256, 512, down=True, act="leaky") # 16x16
        self.e5 = UNetBlock(512, 512, down=True, act="leaky") # 8x8
        
        # Decoder with skip connections
        self.d1 = UNetBlock(512, 512, down=False, act="relu", use_dropout=True)
        self.d2 = UNetBlock(1024, 256, down=False, act="relu", use_dropout=True)
        self.d3 = UNetBlock(512, 128, down=False, act="relu")
        self.d4 = UNetBlock(256, 64, down=False, act="relu")
        
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        
        d1 = self.d1(e5)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        
        out = self.out_conv(torch.cat([d4, e1], dim=1))
        return out

class ThermalToRGBEngine:
    def __init__(self, model_dir=LOCAL_PIX2PIX_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.model = None
        self.is_weights_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load pre-trained weights from local directory or Hugging Face repository."""
        self.model = Pix2PixGenerator().to(self.device)
        self.model.eval()

        pth_file = None
        if self.model_dir.exists():
            pth_candidates = list(self.model_dir.glob("*.pth")) + list(self.model_dir.glob("*.bin")) + list(self.model_dir.glob("*.safetensors"))
            if pth_candidates:
                pth_file = pth_candidates[0]

        if pth_file and pth_file.exists():
            try:
                state_dict = torch.load(pth_file, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.is_weights_loaded = True
                print(f"[Pix2PixEngine] Loaded pre-trained weights from {pth_file} on device: {self.device}")
            except Exception as e:
                print(f"[Pix2PixEngine] Could not load state_dict: {e}. Using structural enhancement fallback.")
        else:
            print(f"[Pix2PixEngine] No weights found in {self.model_dir}. Operating with structural enhancement mode.")

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
