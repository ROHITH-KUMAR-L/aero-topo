# Aero-Topo — Smoke-Resilient UAV Multimodal Perception & 3D Topography System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r160-black.svg)](https://threejs.org/)

**Aero-Topo** is an autonomous UAV wildfire perception system designed to provide dense relative 3D scene geometry and topographical visual representations under severe smoke occlusion, low illumination, and complex forest fire conditions.

---

## Architecture Overview

```text
┌────────────────────────┐      ┌────────────────────────┐
│     Visible RGB        │      │   Thermal IR (TIFF)    │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                  [ Spatial Alignment ]
                            │
                            ▼
                    ┌──────────────┐
                    │  FF-Fusion   │
                    └───────┬──────┘
                            │ (Smoke-Resilient Fused Visual Representation)
                            ▼
                 ┌────────────────────┐
                 │ Depth Anything V2  │
                 └──────────┬─────────┘
                            │ (Dense Relative Depth Map)
                            ▼
              [ Pinhole Camera Projection ]
            X = (u - cx)*Z/fx , Y = (v - cy)*Z/fy
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    ┌──────────────┐                ┌──────────────┐
    │ 3D Pointcloud│                │ Terrain Mesh │
    │   (.PLY)     │                │ (.OBJ/.GLB)  │
    └───────┬──────┘                └───────┬──────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │ Three.js Dashboard │
                 └────────────────────┘
```

---

## Core Technical Concepts

### 1. Project Motivation
Standard RGB cameras provide high visual detail but suffer extreme degradation in wildfire environments due to dense smoke haze, scattering, and low contrast. Thermal Infrared (LWIR/MWIR) sensors penetrate smoke and expose critical ground structures and heat sources. Aero-Topo fuses both modalities to produce a geometrically coherent 3D scene representation.

### 2. Primary Model — FF-Fusion
FF-Fusion (*Knowledge-Distilled Visible-Infrared Image Fusion for Forest Fire Monitoring*) extracts high-frequency structural details from visible RGB and combines them with thermal radiation distributions under heavy smoke without color distortion.

### 3. Depth Model — Depth Anything V2
We utilize **Depth Anything V2 Small** (`vits`, ~24.8M parameters) as our foundation monocular relative depth engine, optimized for edge/RTX 3050 GPUs.

### 4. Relative vs Metric Depth
Standard Depth Anything V2 outputs **relative depth** ($Z \in [0, 1]$ or arbitrary relative scale), not calibrated physical meters. The UI explicitly labels depth as **Relative Depth**.

### 5. 3D Projection Equations
Given depth $Z(u,v)$ and camera parameters $(f_x, f_y, c_x, c_y)$:
$$X = \frac{(u - c_x) \cdot Z}{f_x}$$
$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$
$$Z = Z$$

---

## Setup & Running

### Installation
```bash
git clone https://github.com/ROHITH-KUMAR-L/aero-topo.git
cd aero-topo
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Optionally add `OPENAI_API_KEY` if testing the optional generative comparison branch.

### One-Command Start
```bash
python run.py
```
Open your browser at: `http://127.0.0.1:8000`

---

## Evaluation & Ablation Studies

Run quantitative metric evaluation:
```bash
python scripts/evaluate.py path/to/rgb.png path/to/fused.png
```

---

## License & Attribution

- **FF-Fusion**: Official research framework.
- **Depth Anything V2**: Apache-2.0 License.
- **Aero-Topo**: MIT License.