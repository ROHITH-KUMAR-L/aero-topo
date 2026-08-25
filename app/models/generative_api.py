import os
import logging
import base64
import io
import requests
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("AeroTopo.GenerativeAPI")

DEFAULT_PROMPT = (
    "Create a realistic visible-spectrum interpretation of this thermal image. "
    "Preserve the exact camera viewpoint, scene geometry, terrain layout, vegetation boundaries, "
    "fire locations, and relative spatial arrangement visible in the source thermal image. "
    "Do not invent objects, remove terrain, move fire locations, or alter the camera perspective. "
    "Produce a realistic RGB-style scene interpretation."
)

class GenerativeRGBClient:
    """
    Optional Generative AI Client using OpenAI API (GPT-Image-2 / DALL-E 3).
    Generates synthetic RGB from thermal image for visual comparison in ablation studies.
    NOTE: Output is experimental visualization ONLY and is NEVER used for authoritative 3D geometry.
    """
    def __init__(self, enabled: bool = False, model: str = "gpt-image-2", prompt: str = DEFAULT_PROMPT):
        self.enabled = enabled
        self.model = model
        self.prompt = prompt
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "Generative API disabled in config.yaml."
        if not self.api_key or self.api_key.startswith("your_"):
            return False, "OPENAI_API_KEY environment variable is not set."
        return True, "Generative API ready."

    def generate_synthetic_rgb(self, thermal_3ch: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        available, msg = self.is_available()
        if not available:
            return None, {
                "status": "DISABLED",
                "message": msg,
                "is_authoritative_geometry": False,
                "label": "Generative RGB — Disabled"
            }

        try:
            # Encode thermal image to PNG bytes
            pil_img = Image.fromarray(thermal_3ch)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Call OpenAI API endpoint
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            files = {
                "image": ("thermal.png", img_bytes, "image/png"),
                "prompt": (None, self.prompt),
                "model": (None, self.model),
                "n": (None, "1"),
                "size": (None, "1024x1024")
            }

            logger.info(f"Calling OpenAI image generation API model={self.model}...")
            response = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                res_data = response.json()
                img_url = res_data["data"][0]["url"]
                img_resp = requests.get(img_url, timeout=15)
                gen_pil = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                gen_np = np.array(gen_pil)
                
                # Resize to match thermal image dimensions
                h, w = thermal_3ch.shape[:2]
                gen_resized = cv2.resize(gen_np, (w, h), interpolation=cv2.INTER_CUBIC)

                return gen_resized, {
                    "status": "SUCCESS",
                    "model": self.model,
                    "is_authoritative_geometry": False,
                    "label": "Generative RGB — Experimental (Not Ground Truth)"
                }
            else:
                logger.error(f"OpenAI API Error ({response.status_code}): {response.text}")
                return None, {
                    "status": "API_ERROR",
                    "error_code": response.status_code,
                    "message": response.text,
                    "is_authoritative_geometry": False
                }

        except Exception as e:
            logger.error(f"Generative API exception: {e}")
            return None, {
                "status": "ERROR",
                "message": str(e),
                "is_authoritative_geometry": False
            }
