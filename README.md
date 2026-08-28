# Aero-Topo — Smoke-Resilient Thermal Perception & 3D Topography Workstation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r160-black.svg)](https://threejs.org/)

**Aero-Topo** is a wildfire perception and 3D reconstruction system designed for environments where visible-spectrum imagery is occluded by dense smoke, low illumination, or active fire conditions.

The primary input to the system is a **Thermal / Infrared (IR) image**. The user is **NOT** required to upload an RGB image. The application generates a visible-spectrum representation from the thermal image using a pretrained conditional GAN (cGAN), fuses it with the original thermal image using FF-Fusion, estimates relative depth using Depth Anything V2, and projects a 3D terrain surface for interactive browser visualization.

---

## Authoritative System Architecture

```text
               THERMAL / IR IMAGE (8-bit or 16-bit)
                                │
                                ▼
                      Thermal Preprocessing
                   1-Channel Tensor [1, 256, 256]
                                │
                                ▼
                      Pretrained Pix2Pix cGAN
                     UNetGenerator (FLAME 3)
                                │
                                ▼
                          GENERATED RGB
                       Tensor [3, 256, 256]
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
          Generated RGB                 Original Thermal
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                            FF-Fusion
                                │ (Fused Multimodal Representation)
                                ▼
                        Depth Anything V2
                                │ (Dense Relative Depth Map)
                                ▼
                    [ Camera Pinhole Projection ]
                  X = (u - cx)*Z/fx , Y = (v - cy)*Z/fy
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        3D Pointcloud (.PLY)             Terrain Mesh (.OBJ / .GLB)
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                     Three.js 3D Viewport
```

---

## cGAN Model Synchronization (FLAME 3)

The thermal-to-visible translation model is trained externally on the **FLAME 3** dataset using a Pix2Pix conditional GAN with a **1-channel input (`in_channels=1`) and 3-channel output (`out_channels=3`) UNetGenerator** architecture:

```python
# 1-channel thermal input normalized to [-1, 1]
# Output Tanh activation in [-1, 1] mapped to RGB [0, 255]
UNetGenerator(in_channels=1, out_channels=3)
```

- **Pretrained Checkpoint:** `generator_best.pth`
- **Model Distribution:** Checkpoints are hosted on Hugging Face and downloaded automatically on application startup.
- **Inference-Only:** The production application performs inference only. Training is executed externally.
- **Discriminator:** The discriminator network is training-only and is omitted from production deployment.

---

## Model Checkpoint Management

Missing pretrained model checkpoints (`generator_best.pth`, `ff_fusion.pth`, `depth_anything_v2.pth`) are automatically downloaded into `models/checkpoints/` from Hugging Face:

```bash
python scripts/download_models.py
```

### Centralized Hugging Face Configuration

Model repository locations are configured in `app/config/config.yaml`:

```yaml
models:
  cgan:
    enabled: true
    checkpoint_path: "models/checkpoints/generator_best.pth"
    huggingface:
      repo_id: "YOUR_HF_USERNAME/YOUR_HF_REPOSITORY"
      filename: "generator_best.pth"
```

For private repositories, specify `HF_TOKEN` in `.env`.

---

## Relative Depth & 3D Reconstruction

1. **Depth Anything V2 Small:** Derives dense relative depth ($Z_{rel}$) from the smoke-resilient fused image.
2. **Relative Scale:** Depth outputs represent scale-ambiguous relative depth.
3. **Pinhole Projection:**
   $$X = \frac{(u - c_x) \cdot Z_{rel}}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z_{rel}}{f_y}$$
4. **Three.js Viewer:** Interactive 3D visualization supporting orbit, pan, zoom, point cloud mode, surface mesh mode, wireframe, height exaggeration, and GLB/OBJ export.

---

## Installation & Running

### 1. Installation
```bash
git clone https://github.com/ROHITH-KUMAR-L/aero-topo.git
cd aero-topo
pip install -r requirements.txt
```

### 2. Environment Setup
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 3. Model Downloader
```bash
python scripts/download_models.py
```

### 4. One-Command Application Start
```bash
python run.py
```
Open your browser at: `http://127.0.0.1:8000`

---

## Development & Testing

Run automated tests:
```bash
python -m pytest tests/
```

Run quantitative evaluation comparing Generated RGB to Reference RGB:
```bash
python scripts/evaluate.py path/to/generated_rgb.png path/to/reference_rgb.png
```

---

## License & Attribution

- **FF-Fusion**: Multimodal Image Fusion framework.
- **Depth Anything V2**: Apache-2.0 License.
- **Aero-Topo**: MIT License.