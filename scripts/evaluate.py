import os
import sys
import json
import numpy as np
import cv2
from typing import Dict, Any

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))

def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(img1.shape) == 3 else img1.astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(img2.shape) == 3 else img2.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def compute_edge_preservation(img_orig: np.ndarray, img_fused: np.ndarray) -> float:
    g1 = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY) if len(img_orig.shape) == 3 else img_orig
    g2 = cv2.cvtColor(img_fused, cv2.COLOR_RGB2GRAY) if len(img_fused.shape) == 3 else img_fused

    e1 = cv2.Canny(g1, 50, 150)
    e2 = cv2.Canny(g2, 50, 150)

    intersection = np.logical_and(e1 > 0, e2 > 0)
    union = np.logical_or(e1 > 0, e2 > 0)

    if np.sum(union) == 0:
        return 1.0
    return float(np.sum(intersection) / np.sum(union))

def main():
    print("=== Aero-Topo cGAN & Fusion Quantitative Evaluation ===")
    if len(sys.argv) < 3:
        print("Usage: python scripts/evaluate.py <generated_rgb_path> <reference_rgb_path>")
        return

    rgb_path = sys.argv[1]
    fused_path = sys.argv[2]

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    fused = cv2.cvtColor(cv2.imread(fused_path), cv2.COLOR_BGR2RGB)

    psnr = compute_psnr(rgb, fused)
    ssim = compute_ssim_simple(rgb, fused)
    edge_pres = compute_edge_preservation(rgb, fused)

    results = {
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim, 4),
        "EdgePreservation": round(edge_pres, 4)
    }

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
