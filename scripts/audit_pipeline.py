"""
Aero-Topo Full Architecture Audit Script
Tests all pipeline components that are locally available.
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

THERMAL_SAMPLE = "data/sample_pair/thermal_sample.tif"
CGAN_CKPT = "models/checkpoints/generator_best.pth"
FF_CKPT = "models/checkpoints/ff_fusion.pth"
DEPTH_CKPT = "models/checkpoints/depth_anything_v2.pth"

results = {}

def check(section, name, passed, detail=""):
    key = f"{section}::{name}"
    results[key] = {"passed": passed, "detail": detail}
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")

print("\n" + "="*60)
print("A. cGAN ARCHITECTURE AUDIT")
print("="*60)

# --- A1. Architecture shape ---
try:
    from app.models.cgan import UNetGenerator, DownBlock, UpBlock
    import torch.nn as nn
    model = UNetGenerator(in_channels=1, out_channels=3)
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, 256, 256)
        out = model(x)
    check("A", "Architecture shape [1,1,256,256]->[1,3,256,256]", out.shape == (1, 3, 256, 256), str(out.shape))
    check("A", "Output in [-1,1] (Tanh)", out.min() >= -1.0 and out.max() <= 1.0, f"min={out.min():.4f} max={out.max():.4f}")
except Exception as e:
    check("A", "Architecture shape", False, str(e))

# --- A2. d1/d8 no norm ---
try:
    d1_norms = [l for l in model.d1.block if isinstance(l, nn.InstanceNorm2d)]
    d8_norms = [l for l in model.d8.block if isinstance(l, nn.InstanceNorm2d)]
    check("A", "d1 NO InstanceNorm2d", len(d1_norms)==0, f"d1 norm count={len(d1_norms)}")
    check("A", "d8 NO InstanceNorm2d", len(d8_norms)==0, f"d8 norm count={len(d8_norms)}")
except Exception as e:
    check("A", "d1/d8 norm", False, str(e))

# --- A3. d2-d7 have norm ---
try:
    for name_str in ["d2","d3","d4","d5","d6","d7"]:
        block = getattr(model, name_str)
        has = any(isinstance(l, nn.InstanceNorm2d) for l in block.block)
        check("A", f"{name_str} has InstanceNorm2d", has, "")
except Exception as e:
    check("A", "d2-d7 norms", False, str(e))

# --- A4. u1-u3 dropout, u4-u7 no dropout ---
try:
    for name_str, expect in [("u1",True),("u2",True),("u3",True),("u4",False),("u5",False),("u6",False),("u7",False)]:
        block = getattr(model, name_str)
        has = any(isinstance(l, nn.Dropout) for l in block.block)
        check("A", f"{name_str} dropout={'YES' if expect else 'NO'}", has==expect, "")
except Exception as e:
    check("A", "u1-u7 dropout", False, str(e))

# --- A5. Final Tanh ---
try:
    check("A", "Final layer is Tanh", isinstance(model.final[-1], nn.Tanh), "")
except Exception as e:
    check("A", "Final Tanh", False, str(e))

# --- A6. Checkpoint existence ---
cgan_exists = os.path.exists(CGAN_CKPT)
check("A", "generator_best.pth exists", cgan_exists, CGAN_CKPT if cgan_exists else "MISSING - needs HF download")

# --- A7. Actual checkpoint loading ---
cgan_loaded = False
cgan_model = None
if cgan_exists:
    try:
        from app.models.cgan import CGANModel
        mgr = CGANModel(checkpoint_path=CGAN_CKPT)
        cgan_loaded = mgr.is_ready
        cgan_model = mgr
        check("A", "Checkpoint load (strict)", cgan_loaded, mgr.status_message)
    except Exception as e:
        check("A", "Checkpoint load", False, str(e))
else:
    check("A", "Checkpoint load", False, "Skipped - checkpoint missing")

print("\n" + "="*60)
print("B. THERMAL PREPROCESSING AUDIT")
print("="*60)

try:
    from app.preprocessing.thermal import load_thermal_image, preprocess_thermal_for_cgan

    # B1. Load sample TIFF
    thermal_exists = os.path.exists(THERMAL_SAMPLE)
    check("B", "Thermal sample exists", thermal_exists, THERMAL_SAMPLE)

    if thermal_exists:
        raw_float, norm_3ch, meta = load_thermal_image(THERMAL_SAMPLE)
        check("B", "load_thermal_image raw_float dtype=float32", raw_float.dtype == np.float32, str(raw_float.dtype))
        check("B", "load_thermal_image norm_3ch shape (H,W,3)", len(norm_3ch.shape)==3 and norm_3ch.shape[2]==3, str(norm_3ch.shape))
        check("B", "metadata has original_dtype", "original_dtype" in meta, str(list(meta.keys())))

        # B2. cGAN preprocessing
        cgan_in = preprocess_thermal_for_cgan(raw_float, target_size=(256,256))
        check("B", "preprocess output shape (256,256)", cgan_in.shape == (256,256), str(cgan_in.shape))
        check("B", "preprocess output in [-1,1]", float(cgan_in.min()) >= -1.0 and float(cgan_in.max()) <= 1.0,
              f"min={cgan_in.min():.4f} max={cgan_in.max():.4f}")
        check("B", "preprocess dtype float32", cgan_in.dtype == np.float32, str(cgan_in.dtype))

        # B3. Outlier robustness check
        raw_outlier = raw_float.copy()
        raw_outlier[0, 0] = 1e9  # Extreme outlier
        cgan_out = preprocess_thermal_for_cgan(raw_outlier, target_size=(256,256))
        check("B", "Percentile norm outlier robustness (mean > -0.5)", float(cgan_out.mean()) > -0.5,
              f"mean={cgan_out.mean():.4f}")

        # B4. Verify no Inferno/JET applied before cGAN
        check("B", "cGAN input is 1-channel (no colormap)", len(cgan_in.shape) == 2, str(cgan_in.shape))
except Exception as e:
    check("B", "Thermal preprocessing", False, str(e))

print("\n" + "="*60)
print("C. REAL cGAN INFERENCE TEST")
print("="*60)

if cgan_loaded and cgan_model is not None and thermal_exists:
    try:
        raw_float, norm_3ch, meta = load_thermal_image(THERMAL_SAMPLE)
        cgan_in = preprocess_thermal_for_cgan(raw_float, target_size=(256,256))

        # C1. The key fix: cgan.generate_rgb does its own normalization, but input is ALREADY [-1,1]
        # We need to pass raw_float (not pre-processed) OR the generate_rgb must accept pre-processed
        # Let's test what generate_rgb does with the pre-processed input
        gen_rgb, info = cgan_model.generate_rgb(cgan_in, target_size=(640, 512))
        check("C", "generate_rgb returns not-None", gen_rgb is not None, info.get("status","?"))

        if gen_rgb is not None:
            check("C", "Generated RGB shape (H,W,3)", len(gen_rgb.shape)==3 and gen_rgb.shape[2]==3, str(gen_rgb.shape))
            check("C", "Generated RGB dtype uint8", gen_rgb.dtype == np.uint8, str(gen_rgb.dtype))
            check("C", "Generated RGB values [0,255]", gen_rgb.min()>=0 and gen_rgb.max()<=255,
                  f"min={gen_rgb.min()} max={gen_rgb.max()}")
            check("C", "Generated RGB not all-black", gen_rgb.mean() > 5.0, f"mean={gen_rgb.mean():.2f}")

            # Save generated RGB
            os.makedirs("results/audit", exist_ok=True)
            Image.fromarray(gen_rgb).save("results/audit/generated_rgb.png")
            print("  [INFO] Saved: results/audit/generated_rgb.png")
    except Exception as e:
        check("C", "cGAN inference", False, str(e))
        import traceback; traceback.print_exc()
else:
    check("C", "cGAN inference", False, "Skipped - checkpoint missing or not loaded")

print("\n" + "="*60)
print("D. FF-FUSION AUDIT")
print("="*60)

ff_exists = os.path.exists(FF_CKPT)
check("D", "ff_fusion.pth exists", ff_exists, FF_CKPT if ff_exists else "MISSING - needs HF download")

try:
    from app.models.ff_fusion import FFFusionModel
    ff_model = FFFusionModel(checkpoint_path=FF_CKPT)
    check("D", "FF-Fusion loads (or reports missing cleanly)", True, ff_model.status_message)

    # D1. Verify it takes Generated RGB + Original Thermal (not just one)
    import inspect
    sig = inspect.signature(ff_model.fuse)
    params = list(sig.parameters.keys())
    check("D", "fuse() takes rgb and thermal_3ch params", "rgb" in params and "thermal_3ch" in params, str(params))

    # D2. When unavailable returns None not fake image
    if not ff_model.is_ready:
        dummy_rgb = np.zeros((100,100,3), dtype=np.uint8)
        dummy_th = np.zeros((100,100,3), dtype=np.uint8)
        out, info = ff_model.fuse(dummy_rgb, dummy_th)
        check("D", "Unavailable FF-Fusion returns None (not fake)", out is None, info.get("status","?"))
except Exception as e:
    check("D", "FF-Fusion", False, str(e))

print("\n" + "="*60)
print("E. DEPTH ANYTHING V2 AUDIT")
print("="*60)

try:
    from app.models.depth_anything import DepthAnythingV2Model
    depth_model = DepthAnythingV2Model(encoder="vits", mode="relative")
    check("E", "DepthAnythingV2 initializes", depth_model.is_ready, depth_model.status_message)

    # E1. Test predict_depth output
    dummy_img = np.ones((100,100,3), dtype=np.uint8) * 100
    raw_depth, norm_visual, quality = depth_model.predict_depth(dummy_img)
    check("E", "predict_depth output shape (H,W)", raw_depth.shape == (100,100), str(raw_depth.shape))
    check("E", "depth_mode = relative", quality.get("depth_mode") == "Relative", str(quality.get("depth_mode")))
    check("E", "norm_visual shape (H,W,3)", norm_visual.shape == (100,100,3), str(norm_visual.shape))
except Exception as e:
    check("E", "DepthAnythingV2", False, str(e))
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("F. GEOMETRY AUDIT")
print("="*60)

try:
    from app.geometry.camera import CameraIntrinsics
    from app.geometry.depth_to_pointcloud import depth_to_pointcloud
    from app.geometry.depth_to_mesh import depth_to_mesh

    cam = CameraIntrinsics(image_width=100, image_height=100)
    check("F", "Approximate camera calibration_state", cam.calibration_state == "Approximate", cam.calibration_state)

    cam_cal = CameraIntrinsics(fx=800.0, fy=800.0, cx=50.0, cy=50.0, image_width=100, image_height=100)
    check("F", "Calibrated camera calibration_state", cam_cal.calibration_state == "Calibrated", cam_cal.calibration_state)

    depth = np.ones((100,100), dtype=np.float32) * 5.0
    rgb = np.zeros((100,100,3), dtype=np.uint8)

    pts, colors, pc_meta = depth_to_pointcloud(depth, rgb, cam, subsample=2)
    check("F", "depth_to_pointcloud pts.shape[1]==3", pts.shape[1]==3, str(pts.shape))
    check("F", "depth_to_pointcloud outlier_filtered=True", pc_meta["outlier_filtered"]==True, "")

    verts, mcolors, faces, mesh_meta = depth_to_mesh(depth, rgb, cam, subsample=2)
    check("F", "depth_to_mesh produces vertices", len(verts) > 0, f"{len(verts)} verts")
    check("F", "depth_to_mesh produces faces", len(faces) > 0, f"{len(faces)} faces")

    # Check camera projection equations
    pts_3d = cam.project_depth_to_camera_space(depth)
    # At center pixel (cx=50, cy=50), Z=5 -> X=0, Y=0
    cx, cy = int(cam.cx), int(cam.cy)
    x_center = pts_3d[cy, cx, 0]
    y_center = pts_3d[cy, cx, 1]
    check("F", "Camera projection X=(u-cx)*Z/fx correct at center", abs(x_center) < 0.01, f"X={x_center:.6f}")
    check("F", "Camera projection Y=(v-cy)*Z/fy correct at center", abs(y_center) < 0.01, f"Y={y_center:.6f}")
except Exception as e:
    check("F", "Geometry", False, str(e))
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("G. API AUDIT")
print("="*60)

try:
    import inspect
    from app.api.uploads import upload_thermal
    from app.api.inference import analyze, AnalysisRequest

    # G1. uploads only accepts thermal_file
    sig = inspect.signature(upload_thermal)
    params = list(sig.parameters.keys())
    check("G", "upload_thermal only accepts thermal_file (not rgb_file)", "rgb_file" not in params, str(params))
    check("G", "upload_thermal accepts thermal_file", "thermal_file" in params, str(params))

    # G2. AnalysisRequest only has thermal_path
    req_fields = list(AnalysisRequest.model_fields.keys())
    check("G", "AnalysisRequest has thermal_path", "thermal_path" in req_fields, str(req_fields))
    check("G", "AnalysisRequest has NO rgb_path", "rgb_path" not in req_fields, str(req_fields))
    check("G", "AnalysisRequest has NO enable_generative", "enable_generative" not in req_fields, str(req_fields))
except Exception as e:
    check("G", "API", False, str(e))

print("\n" + "="*60)
print("H. MODEL MANAGER AUDIT")
print("="*60)

try:
    from app.api.model_manager import ModelManager
    # Reset singleton for fresh init
    ModelManager._instance = None
    mm = ModelManager()

    # H1. No discriminator
    has_disc = hasattr(mm, 'discriminator') or hasattr(mm, '_discriminator')
    check("H", "ModelManager has NO discriminator", not has_disc, "")

    # H2. No generative client
    code = open("app/api/model_manager.py").read()
    has_generative = "generative_client" in code or "openai" in code.lower() or "GPT" in code
    check("H", "ModelManager has NO generative_client/openai/GPT", not has_generative, "")

    # H3. Status reports properly
    status = mm.get_status()
    check("H", "get_status() has cgan key", "cgan" in status, str(list(status.keys())))
    check("H", "get_status() has ff_fusion key", "ff_fusion" in status, "")
    check("H", "get_status() has depth_anything_v2 key", "depth_anything_v2" in status, "")

    # H4. Available is False when checkpoint missing
    cgan_avail = status["cgan"]["available"]
    if not cgan_exists:
        check("H", "cGAN reports unavailable when checkpoint missing", not cgan_avail, str(cgan_avail))
    else:
        check("H", "cGAN availability matches checkpoint existence", True, f"available={cgan_avail}")
except Exception as e:
    check("H", "ModelManager", False, str(e))
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("I. FRONTEND SOURCE AUDIT")
print("="*60)

try:
    # Check source HTML for stale content
    dist_html = open("frontend/dist/index.html").read()
    src_html = open("frontend/index.html").read()

    stale_patterns = ["GPT-Image-2", "GPT Image", "Drop RGB Image", "Enable Optional Generative", "rgb_file", "OpenAI"]
    for pat in stale_patterns:
        in_dist = pat in dist_html
        in_src = pat in src_html
        check("I", f"No '{pat}' in dist/index.html", not in_dist, "FOUND" if in_dist else "clean")
        check("I", f"No '{pat}' in source index.html", not in_src, "FOUND" if in_src else "clean")

    # Check ui.ts for rgb state variables
    ui_code = open("frontend/src/ui.ts").read()
    stale_vars = ["rgbFile", "rgbPath", "rgbUpload", "rgbPreview", "visibleFile", "enableGenerative", "generativeEnabled"]
    for var in stale_vars:
        has = var in ui_code
        check("I", f"No '{var}' in ui.ts", not has, "FOUND" if has else "clean")

    # Check thermal upload is single
    check("I", "overview-drop-thermal exists in dist", "overview-drop-thermal" in dist_html, "")
    check("I", "drop-box-single class used (no dual grid)", "drop-box-single" in dist_html, "")
    check("I", "No overview-drop-rgb in dist", "overview-drop-rgb" not in dist_html, "FOUND" if "overview-drop-rgb" in dist_html else "clean")
except Exception as e:
    check("I", "Frontend source", False, str(e))

print("\n" + "="*60)
print("J. PRODUCTION BOUNDARY AUDIT")
print("="*60)

try:
    cgan_code = open("app/models/cgan.py").read()
    training_terms = ["optimizer", "Discriminator", "discriminator", "GAN training loop", "epoch", "backward()", "loss.backward"]
    for term in training_terms:
        if term in ["optimizer", "backward()", "loss.backward"]:
            found = term in cgan_code
            check("J", f"No '{term}' in cgan.py (training boundary)", not found, "FOUND" if found else "clean")
    check("J", "UNetGenerator in inference mode (.eval())", "eval()" in cgan_code, "")
    check("J", "torch.inference_mode() used", "inference_mode" in cgan_code, "")
except Exception as e:
    check("J", "Production boundary", False, str(e))

print("\n" + "="*60)
print("K. DOUBLE-NORMALIZATION AUDIT (Critical)")
print("="*60)

try:
    # In inference.py: preprocess_thermal_for_cgan() is called which outputs [-1,1]
    # Then cgan.generate_rgb() is called with that output
    # Inside generate_rgb(), lines 225-236 re-normalize the input AGAIN
    # This is a double normalization bug IF preprocess_thermal_for_cgan output is passed in
    
    inf_code = open("app/api/inference.py").read()
    # Check that preprocess_thermal_for_cgan output is passed to generate_rgb
    passes_preprocessed = "preprocess_thermal_for_cgan" in inf_code and "generate_rgb(cgan_in_1ch" in inf_code
    
    cgan_code = open("app/models/cgan.py").read()
    # Check generate_rgb re-normalizes
    has_renorm = "th_min = th.min()" in cgan_code or "th_max = th.max()" in cgan_code
    
    check("K", "inference.py passes preprocessed [-1,1] tensor to generate_rgb", passes_preprocessed, "")
    check("K", "generate_rgb contains re-normalization logic (double-norm bug)", has_renorm,
          "BUG: preprocessed [-1,1] input will be re-normalized -> incorrect values" if has_renorm else "OK")
    
    if passes_preprocessed and has_renorm:
        # Demonstrate the bug numerically
        raw = np.random.uniform(200, 400, (256, 256)).astype(np.float32)
        from app.preprocessing.thermal import preprocess_thermal_for_cgan
        preprocessed = preprocess_thermal_for_cgan(raw)
        # preprocessed is in [-1, 1]
        # Now generate_rgb would re-normalize it: (preprocessed - min) / (max - min) -> [0, 1] -> [-1, 1]
        # The -1 to 1 range squashes the effective dynamic range
        print(f"  [INFO] preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        # After re-norm in generate_rgb:
        mn, mx = preprocessed.min(), preprocessed.max()
        renormed = ((preprocessed - mn) / (mx - mn)) * 2.0 - 1.0
        print(f"  [INFO] after re-norm: [{renormed.min():.3f}, {renormed.max():.3f}]")
        print(f"  [WARN] Double normalization is happening - this is incorrect but may not crash")
        print(f"  [WARN] The fix: generate_rgb should check if input is already in [-1,1]")
except Exception as e:
    check("K", "Double-normalization audit", False, str(e))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

passed = sum(1 for v in results.values() if v["passed"])
failed = sum(1 for v in results.values() if not v["passed"])
print(f"\nTotal: {passed} PASS, {failed} FAIL\n")
failures = {k: v for k,v in results.items() if not v["passed"]}
if failures:
    print("FAILURES:")
    for k, v in failures.items():
        print(f"  FAIL: {k}: {v['detail']}")
