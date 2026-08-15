import io
import base64
import numpy as np
from PIL import Image
import cv2

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def read_image_bytes(image_bytes: bytes, filename: str = "") -> np.ndarray:
    """
    Reads raw bytes from an uploaded image (PNG, JPG, TIFF) and converts to 
    a 2D single-channel uint8 numpy array (or 3-channel RGB uint8 array).
    """
    is_tiff = filename.lower().endswith(('.tif', '.tiff'))
    
    if is_tiff and HAS_RASTERIO:
        try:
            with rasterio.open(io.BytesIO(image_bytes)) as src:
                band1 = src.read(1)
                # Normalize float raster band to uint8 [0, 255]
                b_min, b_max = band1.min(), band1.max()
                if b_max > b_min:
                    norm_band = ((band1 - b_min) / (b_max - b_min) * 255.0).astype(np.uint8)
                else:
                    norm_band = np.zeros_like(band1, dtype=np.uint8)
                return norm_band
        except Exception as e:
            print(f"Rasterio read error: {e}, falling back to PIL/OpenCV")

    # Fallback / standard image reading via PIL & OpenCV
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(pil_img)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            # Convert RGBA to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            # Already RGB
            pass
        elif img_np.ndim == 2:
            # Single channel grayscale
            pass
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # Ensure uint8
        if img_np.dtype != np.uint8:
            i_min, i_max = img_np.min(), img_np.max()
            if i_max > i_min:
                img_np = ((img_np - i_min) / (i_max - i_min) * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        return img_np
    except Exception as e:
        raise ValueError(f"Could not parse image format: {e}")

def resize_and_normalize(image_np: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """Resizes an image to target_size (width, height) and ensures 3-channel uint8 representation."""
    if image_np.ndim == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.ndim == 3:
        image_rgb = image_np
    else:
        raise ValueError("Unsupported tensor dimension")

    resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_CUBIC)
    return resized

def numpy_to_base64_data_url(image_np: np.ndarray, is_grayscale: bool = False) -> str:
    """Converts a uint8 numpy array (RGB or Grayscale depth map) to a base64 PNG data URL."""
    if is_grayscale and image_np.ndim == 2:
        pil_img = Image.fromarray(image_np, mode="L")
    elif image_np.ndim == 2:
        # Apply visual colormap to depth/grayscale array for preview if needed, or keep L
        pil_img = Image.fromarray(image_np, mode="L")
    else:
        pil_img = Image.fromarray(image_np, mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
