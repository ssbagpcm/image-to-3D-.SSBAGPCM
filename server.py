#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — single-file, quality-first 2.5D parallax packer + richly documented WebGL viewer.

This file is intentionally, outrageously documented. It includes:
  1) A “packer” pipeline that turns a single input image into a portable .ssbagpcm file (a zip)
     with preview, depth, inpainted background, and layered alpha composites for fallback viewing.
  2) A minimal web server (FastAPI) that serves an embedded HTML viewer (Three.js + custom GLSL).
     The viewer features:
       - Depth-based parallax (foreground moves more than background).
       - Light cinematic depth-of-field (DoF) as a post-process in the fragment shader.
       - Robust “anti-glaze” to avoid ugly stretches at disocclusions by mixing with an
         inpainted background (generated offline with LaMa).
       - Orientation correctness (no upside-down surprises).
       - A thick, full-window overlay black frame that masks any residual stretching that could
         show up at the very edges of the screen, while NOT interfering with user interactions
         (mouse/gyro events still reach the canvas behind).

CLI usage:
  - Generate a pack from an image (no server):
      python server.py your_image.jpg
  - Start the viewer (no generation):
      python server.py --runserver
  - Generate and immediately open the viewer on the result:
      python server.py your_image.jpg --runserver

Installation (Python 3.10+ recommended):
  pip install fastapi uvicorn numpy pillow opencv-python torch huggingface_hub simple-lama-inpainting

Notes:
  - Depth Anything V2: for best results, install its Python module from the official repo.
    If it's not available, we gracefully fallback to MiDaS (torch.hub) without failing.
  - LaMa inpainting: simple-lama-inpainting will auto-download Big-Lama weights at first use.
  - This code prefers QUALITY over speed; you can tune MAX_LONG_SIDE and layer count if needed.

Everything is in English as requested; logs are verbose but human-friendly.
"""

# =============================================================================
# Standard library imports — we keep it simple and explicit
# =============================================================================
import os
import io
import json
import time
import math
import zipfile
import logging
import warnings
import argparse
from typing import Callable, List, Optional, Tuple

# =============================================================================
# Third-party libraries — core building blocks
# =============================================================================
import numpy as np                      # numerical workhorse
from PIL import Image                   # robust image IO and conversions
import cv2                              # OpenCV for filtering/morphology/inpainting
import torch                            # PyTorch for depth models

# FastAPI for serving the viewer and assets; uvicorn as ASGI server
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# URL quoting for safe file parameter passing
from urllib.parse import quote

# HuggingFace Hub — optionally used to fetch Depth Anything V2 weights
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except Exception:
    _HAS_HF = False

# LaMa wrapper — provides a simple Python interface to Big-Lama inpainting
try:
    from simple_lama_inpainting import SimpleLama
    _HAS_LAMA = True
except Exception:
    _HAS_LAMA = False

# =============================================================================
# Global logging + warning hygiene (we keep logs clear and suppress noisy 3rd-party warnings)
# =============================================================================
warnings.filterwarnings("ignore", category=FutureWarning)   # generic deprecations (e.g., timm)
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*", category=UserWarning)

logger = logging.getLogger("ssbagpcm")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(_sh)

# =============================================================================
# Device selection — we pick CUDA if available, else Apple MPS, else CPU
# =============================================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# =============================================================================
# Quality-first defaults — tuned for a clean 2.5D parallax experience
# =============================================================================
# Inference image cap: larger = more detail but slower. 1600 is a good sweet spot.
MAX_LONG_SIDE = 1600

# More layers (4..12) = finer parallax gradations; 7 feels cinematic without huge packs.
DEFAULT_LAYERS = 7

# Default parallax amplitude (fraction of min(width,height)); boosted x1.35 in the viewer for punch.
DEFAULT_PARALLAX = 0.065

# Alpha feather (Gaussian sigma) on each layer's mask – reduces aliasing at blend boundaries.
DEFAULT_BLUR_SIGMA = 1.0

# Layer separation in near-depth space (near=1, far=0): RBF sigma; lower = crisper separation.
SEPARATION_SIGMA = 0.080

# Bias for layer centers: exponent < 1 → more centers near the foreground (stronger foreground motion).
NEAR_BIAS_GAMMA = 0.80

# Depth smoothing settings — reduces noise and halos in raw depth predictions.
DEPTH_SMOOTH = True
DEPTH_SMOOTH_RADIUS = 12
DEPTH_SMOOTH_EPS = 1e-3

# Disocclusion mask params — how we detect where to inpaint to avoid “glaze/stretch”.
BG_EDGE_THRESHOLD = 0.030       # base depth gradient threshold
BG_DILATE_MAX = 196             # max dilation radius in pixels (scaled with parallax)
BG_INPAINT_RADIUS = 12          # Telea radius if LaMa is unavailable
BG_EXTRA_ERODE = 0              # not used by default; can reduce over-dilation if set > 0
LAMA_EXTRA_DILATE = 3           # slight pre-dilate before LaMa to avoid hairline seams
LAMA_STRIDE = 8                 # pad H/W to multiple-of-8 for LaMa stability; crop back after

# =============================================================================
# Image utilities — robust RGB↔BGR conversions, resizing, normalization, enhancement
# =============================================================================
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """
    Convert a PIL.Image (RGB) to an OpenCV ndarray (BGR, uint8).
    We standardize to BGR when using OpenCV APIs.
    """
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV ndarray (BGR) back to a PIL.Image (RGB).
    """
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def resize_max_side(img_pil: Image.Image, max_side: int = MAX_LONG_SIDE) -> Image.Image:
    """
    Fit image within a maximum long side while preserving aspect ratio.
    If already smaller, do nothing (we prefer not to upscale for depth).
    """
    w, h = img_pil.size
    m = max(w, h)
    if m <= max_side:
        return img_pil
    scale = max_side / float(m)
    new_size = (int(w * scale), int(h * scale))
    return img_pil.resize(new_size, Image.LANCZOS)

def normalize01(x: np.ndarray) -> np.ndarray:
    """
    Normalize any numeric array to [0..1] safely.
    """
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def enhance_image(img_pil: Image.Image, sharp_strength: float = 0.6, clahe_clip: float = 2.0, denoise: bool = True) -> Image.Image:
    """
    A fast, quality-biased enhancement routine:
      - bilateral denoise to preserve edges while smoothing color noise,
      - unsharp mask (via addWeighted + Gaussian blur),
      - CLAHE on the L channel in LAB space to lift micro-contrast gently.
    Returns a new PIL image (RGB).
    """
    img = pil_to_cv(img_pil)
    if denoise:
        img = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.addWeighted(img, 1.0 + sharp_strength, blur, -sharp_strength, 0.0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv_to_pil(img2)

# =============================================================================
# Depth estimation — Depth Anything V2 (preferred) + MiDaS (fusion / fallback)
# =============================================================================
_DAV2_MODEL = None
_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None

def _disable_sdp_backends():
    """
    Depth Anything V2 may try to use flash/memory-efficient attention backends that
    are not supported on CPU. On CPU-only hosts this raises runtime errors like:
      "No operator found for memory_efficient_attention_forward..."
    We proactively disable those and force math SDP.
    """
    os.environ.setdefault("PYTORCH_SDP_DISABLE_BACKENDS", "flash,mem_efficient")
    try:
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass  # If backend toggling isn't available, we silently ignore it.

def load_depth_anything_v2():
    """
    Load Depth Anything V2 (best quality/speed combo) from its Python module.
    We download the weights via HuggingFace Hub; this requires a working internet connection.
    If the module isn't installed, we return None so the pipeline can fallback to MiDaS.
    """
    global _DAV2_MODEL
    if _DAV2_MODEL is not None:
        return _DAV2_MODEL
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        logger.warning("Depth Anything V2 module not found. Install the repo for best results. Error: %s", e)
        return None

    if not _HAS_HF:
        logger.error("huggingface_hub not available — cannot fetch DAV2 weights.")
        return None

    candidates = [
        ("vitl", "depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"),
        ("vitb", "depth-anything/Depth-Anything-V2-Base",  "depth_anything_v2_vitb.pth"),
        ("vits", "depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
    ]
    for enc, repo, fname in candidates:
        try:
            logger.info("Downloading DAV2 weights: %s/%s", repo, fname)
            weights_path = hf_hub_download(repo_id=repo, filename=fname, repo_type="model")
            from depth_anything_v2.dpt import DepthAnythingV2
            feats = {"vitl": 256, "vitb": 128, "vits": 64}.get(enc, 256)
            out_ch = {"vitl": [256, 512, 1024, 1024], "vitb": [96, 192, 384, 768], "vits": [48, 96, 192, 384]}.get(enc, [256, 512, 1024, 1024])
            model = DepthAnythingV2(encoder=enc, features=feats, out_channels=out_ch)
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE).eval()
            _DAV2_MODEL = model
            logger.info("Depth Anything V2 ready (%s) on %s", enc, DEVICE)
            return _DAV2_MODEL
        except Exception as e:
            logger.warning("Failed to load DAV2 (%s): %s", repo, e)
    logger.error("All DAV2 candidates failed — will rely on MiDaS.")
    return None

def load_midas():
    """
    Load MiDaS DPT_Large via torch.hub; it auto-downloads models/transforms if not cached.
    Used both as a standalone fallback and as a fusion partner (median after quantile matching).
    """
    global _MIDAS_MODEL, _MIDAS_TRANSFORM
    if _MIDAS_MODEL is not None:
        return _MIDAS_MODEL, _MIDAS_TRANSFORM
    try:
        try:
            _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        except TypeError:
            _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        _MIDAS_MODEL.to(DEVICE).eval()
        try:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        except TypeError:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _MIDAS_TRANSFORM = transforms.dpt_transform
        logger.info("MiDaS DPT_Large ready on %s", DEVICE)
        return _MIDAS_MODEL, _MIDAS_TRANSFORM
    except Exception as e:
        logger.warning("MiDaS unavailable: %s", e)
        return None, None

def quantile_match(src: np.ndarray, ref: np.ndarray, qs: int = 41) -> np.ndarray:
    """
    Align src's value distribution to ref's via piecewise linear mapping on quantiles.
    This is robust to scale/shifts across different depth predictors.
    """
    q = np.linspace(0, 1, qs)
    src_q = np.quantile(src, q)
    ref_q = np.quantile(ref, q)
    flat = src.ravel()
    mapped = np.interp(flat, src_q, ref_q).reshape(src.shape)
    return mapped

def fuse_depths(depths: List[np.ndarray]) -> np.ndarray:
    """
    Fuse multiple depth predictions:
      - Align all to the first via quantile matching.
      - Take median across predictors (reduces outliers).
      - Normalize to [0..1], where far=1, near=0.
    """
    ref = depths[0]
    aligned = [ref] + [quantile_match(d, ref) for d in depths[1:]]
    fused = np.median(np.stack(aligned, axis=0), axis=0).astype(np.float32)
    return normalize01(fused)

@torch.no_grad()
def estimate_depth(img_pil: Image.Image, enable_fusion: bool = True) -> np.ndarray:
    """
    End-to-end depth estimation with quality-first defaults.
    Returns a float32 array (H,W) normalized to [0..1], where far=1, near=0.
    Steps:
      1) Resize to MAX_LONG_SIDE for inference.
      2) Run Depth Anything V2 if available (with CPU-safe SDP backends).
      3) Run MiDaS too (fusion partner or fallback).
      4) Fuse predictions (median after quantile match) if multiple available.
      5) Depth smoothing (guided/joint-bilateral) + slight “unsharp” refine.
    """
    t0 = time.time()
    img_infer = resize_max_side(img_pil, MAX_LONG_SIDE)
    preds = []

    # Prefer DAV2
    dav2 = load_depth_anything_v2()
    if dav2 is not None:
        try:
            _disable_sdp_backends()
            d = dav2.infer_image(cv2.cvtColor(np.array(img_infer.convert("RGB")), cv2.COLOR_RGB2BGR)).astype(np.float32)
            if img_infer.size != img_pil.size:
                d = cv2.resize(d, img_pil.size, interpolation=cv2.INTER_CUBIC)
            preds.append(d)
            logger.info("DAV2 depth done (%.1f ms)", (time.time() - t0) * 1000.0)
        except Exception as e:
            logger.warning("DAV2 inference error: %s", e)

    # MiDaS fusion/fallback
    model, tfm = load_midas()
    if model is not None and tfm is not None:
        try:
            img_np = np.array(img_infer.convert("RGB"), dtype=np.uint8)
            sample = tfm(img_np)
            input_batch = sample["image"] if isinstance(sample, dict) else sample
            if isinstance(input_batch, torch.Tensor) and input_batch.ndim == 3:
                input_batch = input_batch.unsqueeze(0)
            input_batch = input_batch.to(DEVICE)
            pred = model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_infer.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze().detach().cpu().numpy().astype(np.float32)
            if img_infer.size != img_pil.size:
                pred = cv2.resize(pred, img_pil.size, interpolation=cv2.INTER_CUBIC)
            preds.append(pred)
            logger.info("MiDaS depth done (%.1f ms)", (time.time() - t0) * 1000.0)
        except Exception as e:
            logger.warning("MiDaS error: %s", e)

    if len(preds) == 0:
        raise RuntimeError("No depth predictor available (DAV2 and MiDaS both failed).")

    # Fuse if multiple predictors succeeded; else use the single one.
    d_norm_far = fuse_depths(preds) if (enable_fusion and len(preds) >= 2) else normalize01(preds[0])

    # Quality refine: guided smoothing + minor unsharp to separate planes a bit more
    if DEPTH_SMOOTH:
        d_norm_far = smooth_depth_guided(img_pil, d_norm_far)
        d_norm_far = refine_depth_details(img_pil, d_norm_far)
    return d_norm_far

def smooth_depth_guided(img_pil: Image.Image, depth_far01: np.ndarray) -> np.ndarray:
    """
    Edge-aware smoothing pass for the depth map:
      - Try guidedFilter (OpenCV ximgproc) using image luminance as guide.
      - Else try jointBilateralFilter (ximgproc).
      - Else fall back to a plain bilateral on the scalar depth.
    This reduces noisy contours and jaggies that would cause wobble in parallax.
    """
    d = np.clip(depth_far01.astype(np.float32), 0.0, 1.0)
    img_bgr = pil_to_cv(img_pil)
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            d_sm = cv2.ximgproc.guidedFilter(guide=guide, src=d, radius=DEPTH_SMOOTH_RADIUS, eps=DEPTH_SMOOTH_EPS)
            return np.clip(d_sm, 0.0, 1.0).astype(np.float32)
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
            d_sm = cv2.ximgproc.jointBilateralFilter(guide=img_bgr, src=d, d=-1, sigmaColor=0.05, sigmaSpace=DEPTH_SMOOTH_RADIUS)
            return np.clip(d_sm, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        logger.warning("guided/joint bilateral error: %s", e)
    d_sm = cv2.bilateralFilter(d, d=9, sigmaColor=0.08, sigmaSpace=float(DEPTH_SMOOTH_RADIUS))
    return np.clip(d_sm, 0.0, 1.0).astype(np.float32)

def refine_depth_details(img_pil: Image.Image, depth_far01: np.ndarray, amount: float = 0.18) -> np.ndarray:
    """
    Gentle “unsharp mask” on the depth to enhance separation between planes:
    d' = clamp(d + amount * (d - blur(d)), 0, 1).
    Too much sharpening can create artifacts; 0.18 is subtle yet helpful.
    """
    d = np.clip(depth_far01.astype(np.float32), 0.0, 1.0)
    blur = cv2.GaussianBlur(d, (0, 0), 1.5)
    d2 = np.clip(d + amount * (d - blur), 0.0, 1.0)
    return d2.astype(np.float32)

# =============================================================================
# LaMa inpainting — robust pad-to-multiple + crop-back + fallback Telea
# =============================================================================
_LAMA = None
def get_lama():
    """
    Lazy-initialize LaMa (simple-lama-inpainting). On first call, this may download weights.
    We normalize return types (PIL or ndarray) downstream to always use uint8 RGB ndarray.
    """
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    if not _HAS_LAMA:
        return None
    try:
        _LAMA = SimpleLama()
        logger.info("LaMa (simple-lama-inpainting) ready.")
        return _LAMA
    except Exception as e:
        logger.warning("Failed to init LaMa: %s", e)
        return None

def _ensure_u8_3ch_rgb(x):
    """
    Ensure an image is uint8 RGB ndarray, regardless of whether it was PIL or grayscale.
    """
    if isinstance(x, Image.Image):
        x = np.array(x.convert("RGB"))
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def _pad_to_multiple(arr: np.ndarray, multiple: int, is_mask: bool = False):
    """
    Pad array to nearest multiple-of-N on both H and W with symmetrical padding.
    Masks use constant 0 padding; images use REFLECT_101 to reduce seam artifacts.
    Returns (padded_array, (top,bottom,left,right)).
    """
    H, W = arr.shape[:2]
    H2 = int(math.ceil(H / multiple) * multiple)
    W2 = int(math.ceil(W / multiple) * multiple)
    pad_h = H2 - H
    pad_w = W2 - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    borderType = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    if arr.ndim == 2:
        out = cv2.copyMakeBorder(arr, top, bottom, left, right, borderType, value=0)
    else:
        out = cv2.copyMakeBorder(arr, top, bottom, left, right, borderType, value=(0,0,0))
    return out, (top, bottom, left, right)

def _crop_from_pad(arr: np.ndarray, pads: Tuple[int, int, int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Crop a padded array back to original target (H,W) using stored pad sizes.
    """
    top, bottom, left, right = pads
    Ht, Wt = target_hw
    H, W = arr.shape[:2]
    y0 = top
    x0 = left
    y1 = min(y0 + Ht, H)
    x1 = min(x0 + Wt, W)
    return arr[y0:y1, x0:x1, ...] if arr.ndim == 3 else arr[y0:y1, x0:x1]

def inpaint_lama_or_telea(img_bgr: np.ndarray, mask_u8: np.ndarray, extra_dilate: int = LAMA_EXTRA_DILATE) -> np.ndarray:
    """
    Run inpainting on (BGR image, binary mask).
    Steps:
      - Binary mask (0/255), optional dilation to cover seam risk.
      - Pad both image and mask to multiple-of-8 (LaMa stability).
      - Run LaMa (RGB) if available, normalize types, convert back to BGR.
      - Else fallback to OpenCV Telea.
      - Crop back to original size (handles any LaMa padding).
    Returns BGR ndarray (uint8), aligned to the input size.
    """
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    mask_bin = (mask_u8 > 127).astype(np.uint8) * 255
    if extra_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_dilate*2+1, extra_dilate*2+1))
        mask_bin = cv2.dilate(mask_bin, k, iterations=1)

    H, W = img_bgr.shape[:2]
    img_pad, pads = _pad_to_multiple(img_bgr, LAMA_STRIDE, is_mask=False)
    mask_pad, _ = _pad_to_multiple(mask_bin, LAMA_STRIDE, is_mask=True)

    lama = get_lama()
    if lama is not None:
        try:
            rgb_pad = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
            rgb_pad = _ensure_u8_3ch_rgb(rgb_pad)
            out_rgb = lama(rgb_pad, mask_pad)  # may return PIL or ndarray
            out_rgb = _ensure_u8_3ch_rgb(out_rgb)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("LaMa inpainting failed, fallback to Telea: %s", e)
            out_bgr = cv2.inpaint(img_pad, mask_pad, BG_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    else:
        out_bgr = cv2.inpaint(img_pad, mask_pad, BG_INPAINT_RADIUS, cv2.INPAINT_TELEA)

    out_crop = _crop_from_pad(out_bgr, pads, (H, W))
    if out_crop.shape[:2] != (H, W):
        out_crop = cv2.resize(out_crop, (W, H), interpolation=cv2.INTER_CUBIC)
    return out_crop

# =============================================================================
# Disocclusion mask — where to inpaint to avoid stretched pixels during parallax
# =============================================================================
def build_disocclusion_mask(img_pil: Image.Image, depth_far01: np.ndarray, parallax: float) -> np.ndarray:
    """
    Construct a conservative (worst-case) disocclusion mask:
      - Compute depth gradient magnitude (Sobel) on near-depth (near=1, far=0).
      - Threshold using a robust percentile heuristic; foreground edges weighted more (near>0.35).
      - Dilate strongly, scaled by parallax amplitude and image size.
      - Morphological closing to fill small holes, slight blur to soften for LaMa.
    Returns an 8-bit single-channel mask (0..255), where 255 indicates “to inpaint”.
    """
    img_bgr = pil_to_cv(img_pil)
    H, W = img_bgr.shape[:2]
    d = np.clip(depth_far01.astype(np.float32), 0.0, 1.0)
    near = 1.0 - d

    gx = cv2.Sobel(near, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(near, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    thr = max(BG_EDGE_THRESHOLD, float(np.percentile(grad, 80)) * 0.55)
    edges = (grad > thr).astype(np.uint8) * 255

    near_mask = (near > 0.35).astype(np.uint8) * 255
    edges_near = cv2.bitwise_and(edges, near_mask)

    minDim = min(W, H)
    base = int(min(BG_DILATE_MAX, max(3, parallax * minDim * 1.95)))
    extra = max(2, int(base * 0.60))
    if base % 2 == 0: base += 1
    if extra % 2 == 0: extra += 1

    k_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base, base))
    k_extra = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra, extra))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, base//3)|1, max(3, base//3)|1))

    mask1 = cv2.dilate(edges, k_base, iterations=1)
    mask2 = cv2.dilate(edges_near, k_extra, iterations=1)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    if BG_EXTRA_ERODE > 0:
        k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_EXTRA_ERODE*2+1, BG_EXTRA_ERODE*2+1))
        mask = cv2.erode(mask, k_er, iterations=1)

    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.6)
    mask = np.clip(mask * 1.35, 0, 255).astype(np.uint8)
    return mask

def make_bgfill(img_pil: Image.Image, depth_far01: np.ndarray, parallax: float) -> Image.Image:
    """
    Build the inpainted background and softly blend with original to hide seams:
      - Detect disocclusions via depth gradients & near weighting.
      - Inpaint using LaMa (preferred) or Telea (fallback).
      - Apply a Gaussian feathered blend (mask blurred & slightly intensified) to avoid hard joins.
    Returns a PIL RGB image (the “bgfill”).
    """
    img_bgr = pil_to_cv(img_pil)
    mask = build_disocclusion_mask(img_pil, depth_far01, parallax)
    inpainted = inpaint_lama_or_telea(img_bgr, mask, extra_dilate=LAMA_EXTRA_DILATE)

    mask_f = (mask.astype(np.float32) / 255.0)
    mask_f = cv2.GaussianBlur(mask_f, (0, 0), 6.0)
    mask_f = np.clip(mask_f * 1.25, 0.0, 1.0)[..., None]

    out = (inpainted.astype(np.float32) * mask_f + img_bgr.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
    return cv_to_pil(out)

# =============================================================================
# Layering — near-biased RBF soft assignment (foreground layers get more motion)
# =============================================================================
def near_biased_centers(num_layers: int, gamma: float = NEAR_BIAS_GAMMA) -> np.ndarray:
    """
    Compute L centers in near-depth space [0..1] (near=1). A gamma < 1 biases centers toward near=1.
    """
    t = np.linspace(0.0, 1.0, num_layers)
    return np.power(t, gamma)

def layers_from_depth_soft(img_pil: Image.Image,
                           depth_far01: np.ndarray,
                           num_layers: int = DEFAULT_LAYERS,
                           rbf_sigma: float = SEPARATION_SIGMA,
                           edge_feather_sigma: float = DEFAULT_BLUR_SIGMA
                           ) -> Tuple[List[Image.Image], List[float]]:
    """
    Construct RGBA layers by soft-assigning pixels via Gaussian RBFs in near-depth space.
    Alpha masks are gently feathered to avoid halos; RGB is always the original image.
    Returns:
      layers: list of PIL RGBA images, one per layer (ordered far->near by “depth_factors”).
      depth_factors: list of floats (0..1) indicating per-layer “far-ness” used by 2D fallback viewers.
    """
    img_rgb = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    d_norm = np.clip(depth_far01.astype(np.float32), 0.0, 1.0)
    near = 1.0 - d_norm

    centers = near_biased_centers(num_layers)
    depth_factors = [(1.0 - float(c)) for c in centers]  # convert near-centers to far factors

    weights_list = []
    for ci in centers:
        w = np.exp(-((near - ci) ** 2) / (2.0 * rbf_sigma * rbf_sigma)).astype(np.float32)
        weights_list.append(w)
    weights = np.stack(weights_list, axis=-1)
    s = np.sum(weights, axis=-1, keepdims=True) + 1e-6
    weights = weights / s

    layers = []
    for i in range(num_layers):
        alpha = weights[..., i]
        if edge_feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), edge_feather_sigma)
        a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        rgba = np.dstack([img_rgb, a8])
        layers.append(Image.fromarray(rgba, "RGBA"))
    return layers, depth_factors

# =============================================================================
# Depth → RG16 packed PNG (two 8-bit channels in RGB)
# =============================================================================
def depth_to_rg16_png(depth_far01: np.ndarray) -> Image.Image:
    """
    Pack a float depth map [0..1] into 16-bit (big-endian): hi 8 bits in R, lo 8 bits in G, B=0.
    Viewer reconstructs: depth = (R*256 + G)/65535.
    """
    d = np.clip(depth_far01, 0.0, 1.0)
    v = np.round(d * 65535.0).astype(np.uint16)
    hi = (v >> 8).astype(np.uint8)
    lo = (v & 255).astype(np.uint8)
    rgb = np.stack([hi, lo, np.zeros_like(hi, dtype=np.uint8)], axis=-1)
    return Image.fromarray(rgb, "RGB")

# =============================================================================
# .ssbagpcm packaging — a simple zip with a clear manifest
# =============================================================================
def write_ssbagpcm(out_path: str,
                   base_preview_pil: Image.Image,
                   layers_rgba: List[Image.Image],
                   depth_factors: List[float],
                   parallax: float,
                   depth_rg16_img: Optional[Image.Image],
                   bgfill_img: Optional[Image.Image],
                   extra_meta: Optional[dict] = None):
    """
    Produce a .ssbagpcm zip with:
      - manifest.json: metadata for rendering/parallax on the client side,
      - preview.png: the base RGB image,
      - depth.png: RG16 packed version of depth (if available),
      - bgfill.png: inpainted RGB background (if available),
      - layers/layer_XX.png: RGBA layers for 2D fallback renderers.

    The viewer uses:
      - preview.png + depth.png (required for depth warp mode),
      - bgfill.png for anti-glaze,
      - manifest fields for dimensions and default parallax.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        w, h = base_preview_pil.size
        manifest = {
            "version": 5,
            "width": w,
            "height": h,
            "num_layers": len(layers_rgba),
            "depth_factors": depth_factors,
            "parallax_default": parallax,
            "generator": "server.py",
            "has_depth": depth_rg16_img is not None,
            "depth_format": "RG16_u8x2",
            "depth_value": "far_is_1",
            "has_bgfill": bgfill_img is not None,
            "has_inpaint": bgfill_img is not None
        }
        if extra_meta:
            manifest.update(extra_meta)
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        buf = io.BytesIO()
        base_preview_pil.save(buf, format="PNG", optimize=True)
        zf.writestr("preview.png", buf.getvalue())

        if depth_rg16_img is not None:
            buf = io.BytesIO()
            depth_rg16_img.save(buf, format="PNG", optimize=True)
            zf.writestr("depth.png", buf.getvalue())

        if bgfill_img is not None:
            buf = io.BytesIO()
            bgfill_img.save(buf, format="PNG", optimize=True)
            zf.writestr("bgfill.png", buf.getvalue())

        for i, im in enumerate(layers_rgba):
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            zf.writestr(f"layers/layer_{i:02d}.png", buf.getvalue())

# =============================================================================
# Full pipeline entry — this is what the CLI calls to build a pack from an image
# =============================================================================
def process_image_to_ssbagpcm(input_path: str,
                              output_path: str,
                              layers: int = DEFAULT_LAYERS,
                              blur_sigma: float = DEFAULT_BLUR_SIGMA,
                              parallax: float = DEFAULT_PARALLAX,
                              enable_fusion: bool = True,
                              progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """
    Orchestrate the entire build:
      1) Load and lightly enhance the image.
      2) Depth estimation (DAV2 + MiDaS fusion if available).
      3) Depth smoothing and detail refine.
      4) Disocclusion mask and inpainting (LaMa preferred).
      5) Pack depth into RG16 PNG for the viewer.
      6) Build soft RBF-based layers for 2D fallback.
      7) Write a .ssbagpcm file with everything inside.

    Returns a dict with path, dimensions, layer count, parallax, and elapsed time.
    """
    t0 = time.time()
    def report(p: float, msg: str):
        if progress:
            progress(float(max(0.0, min(1.0, p))), msg)
        logger.info("%s (%.1f%%)", msg, p * 100)

    report(0.02, "Loading input image")
    img = Image.open(input_path).convert("RGB")
    w0, h0 = img.size
    meta = {"source_file": os.path.abspath(input_path), "source_w": w0, "source_h": h0, "device": str(DEVICE)}

    report(0.08, "Enhancing image (denoise + sharp + CLAHE)")
    enhanced = enhance_image(img)

    report(0.35, "Estimating depth (Depth Anything V2 + MiDaS fusion if available)")
    d_norm_far = estimate_depth(enhanced, enable_fusion=enable_fusion)

    report(0.45, "Smoothing depth (guided/bilateral) + detail refine")
    d_norm_far = smooth_depth_guided(enhanced, d_norm_far)
    d_norm_far = refine_depth_details(enhanced, d_norm_far)

    report(0.62, "Building disocclusion mask + inpainting (LaMa/Telea)")
    bgfill_img = make_bgfill(enhanced, d_norm_far, parallax=parallax)

    report(0.72, "Packing depth to RG16 (PNG RGB)")
    depth_rg16 = depth_to_rg16_png(d_norm_far)

    report(0.86, "Layerizing with stronger separation (soft RBF near-biased)")
    layers_rgba, depth_factors = layers_from_depth_soft(
        enhanced, d_norm_far, num_layers=layers, rbf_sigma=SEPARATION_SIGMA, edge_feather_sigma=blur_sigma
    )

    report(0.94, "Writing .ssbagpcm")
    write_ssbagpcm(output_path, enhanced, layers_rgba, depth_factors, parallax=parallax,
                   depth_rg16_img=depth_rg16, bgfill_img=bgfill_img, extra_meta=meta)

    dt = time.time() - t0
    report(1.0, f"Done in {dt:.2f}s")
    return {
        "output": os.path.abspath(output_path),
        "width": enhanced.size[0],
        "height": enhanced.size[1],
        "layers": layers,
        "parallax": parallax,
        "time_sec": dt
    }

# =============================================================================
# Web server (FastAPI) — serves the viewer and unpacks assets on demand
# =============================================================================
app = FastAPI(title="SSBAGPCM Viewer (Single-file, richly documented)")

def load_manifest_from_zip(path):
    """
    Open a .ssbagpcm zip and parse its manifest; add derived fields for compatibility
    with older packs (has_depth/has_bgfill).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError("manifest.json missing")
        data = json.loads(zf.read("manifest.json").decode("utf-8"))
        if "has_depth" not in data:
            data["has_depth"] = "depth.png" in zf.namelist()
            if data["has_depth"]:
                data["depth_format"] = "RG16_u8x2"
                data["depth_value"] = "far_is_1"
        if "has_bgfill" not in data:
            data["has_bgfill"] = "bgfill.png" in zf.namelist()
        return data

@app.get("/", response_class=HTMLResponse)
def index(file: str = ""):
    """
    Serve the embedded HTML viewer. If ?file=/absolute/path/to/pack.ssbagpcm is provided,
    the viewer auto-loads it immediately via /api/manifest.
    """
    return HTMLResponse(INDEX_HTML)

@app.get("/api/manifest", response_class=JSONResponse)
def api_manifest(file: str):
    """
    Return the parsed manifest plus signed URLs to preview/depth/bg/layers endpoints for the browser.
    """
    try:
        mf = load_manifest_from_zip(file)
        layer_urls = [f"/api/layer?file={quote(file)}&i={i}" for i in range(mf.get("num_layers", 0))]
        preview_url = f"/api/preview?file={quote(file)}"
        depth_url = None
        bg_url = None
        with zipfile.ZipFile(file, "r") as zf:
            if "depth.png" in zf.namelist():
                depth_url = f"/api/depth?file={quote(file)}"
            if "bgfill.png" in zf.namelist():
                bg_url = f"/api/bgfill?file={quote(file)}"
        return JSONResponse({**mf,
                             "layer_urls": layer_urls,
                             "preview_url": preview_url,
                             "depth_url": depth_url,
                             "bgfill_url": bg_url,
                             "file": file})
    except Exception as e:
        raise HTTPException(400, f"manifest error: {e}")

@app.get("/api/preview")
def api_preview(file: str):
    """
    Fetch preview.png bytes from the pack (PNG, sRGB).
    """
    try:
        with zipfile.ZipFile(file, "r") as zf:
            if "preview.png" not in zf.namelist():
                raise ValueError("preview.png missing")
            data = zf.read("preview.png")
            return Response(content=data, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"preview error: {e}")

@app.get("/api/depth")
def api_depth(file: str):
    """
    Fetch depth.png bytes from the pack (PNG, RG16 packed in RGB).
    """
    try:
        with zipfile.ZipFile(file, "r") as zf:
            if "depth.png" not in zf.namelist():
                raise ValueError("depth.png missing")
            data = zf.read("depth.png")
            return Response(content=data, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"depth error: {e}")

@app.get("/api/bgfill")
def api_bgfill(file: str):
    """
    Fetch bgfill.png bytes (PNG, RGB inpainted background).
    """
    try:
        with zipfile.ZipFile(file, "r") as zf:
            if "bgfill.png" not in zf.namelist():
                raise ValueError("bgfill.png missing")
            data = zf.read("bgfill.png")
            return Response(content=data, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"bgfill error: {e}")

@app.get("/api/layer")
def api_layer(file: str, i: int):
    """
    Fetch layers/layer_XX.png bytes (PNG, RGBA). Useful for 2D fallback viewers.
    """
    try:
        with zipfile.ZipFile(file, "r") as zf:
            name = f"layers/layer_{i:02d}.png"
            if name not in zf.namelist():
                raise ValueError(f"missing layer: {name}")
            data = zf.read(name)
            return Response(content=data, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"layer error: {e}")

def start_server(port: int):
    """
    Blocking uvicorn run on localhost:port. Intended to be called by the CLI entry.
    """
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

# =============================================================================
# Embedded HTML viewer — single string served at GET /
#  - Fullscreen WebGL canvas behind everything
#  - Overlay thick black frame masking edges
#  - DOF, anti-stretch, anti-glaze, parallax
#  - Drag & Drop + Open button + Fullscreen
# =============================================================================
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SSBAGPCM Viewer (Three.js ES Modules + overlay frame)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --framePx: clamp(32px, 5.5vw, 96px); }
    html, body { margin:0; height:100%; overflow:hidden; background:#000; color:#e8eaed; font-family:system-ui, sans-serif; }
    #overlayFrame {
      position:fixed; inset:0; border: var(--framePx) solid #000; box-sizing: border-box;
      border-radius: 12px; pointer-events:none; z-index:5;
    }
    #ui {
      position:fixed; left:calc(var(--framePx) + 12px); top:calc(var(--framePx) + 12px);
      display:flex; gap:8px; align-items:center; background:rgba(15,20,28,.6);
      border:1px solid rgba(255,255,255,.08); padding:8px 10px; border-radius:10px;
      backdrop-filter: blur(6px); z-index:10;
    }
    .btn { background:#1f2633; color:#e8eaed; border:1px solid #2a3344; padding:7px 10px; border-radius:8px; cursor:pointer; }
    .btn:hover { background:#2a3344; }
    #fileName { color:#8ab4ff; font-size:12px; max-width:42vw; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    #drop { position:fixed; inset:0; display:none; align-items:center; justify-content:center; border:2px dashed #2a3344; color:#a9b4c7; font-size:18px; pointer-events:none; z-index:8; }
    #hint {
      position:fixed; right:calc(var(--framePx) + 12px); bottom:calc(var(--framePx) + 12px);
      color:#a9b4c7; font-size:12px; opacity:.85; background:rgba(15,20,28,.55);
      border:1px solid rgba(255,255,255,.08); padding:6px 8px; border-radius:8px; backdrop-filter: blur(6px); z-index:10;
    }
  </style>
</head>
<body>
  <div id="overlayFrame"></div>
  <div id="ui">
    <button id="openBtn" class="btn">Open .ssbagpcm</button>
    <input id="fileInput" type="file" accept=".ssbagpcm" style="display:none" />
    <button id="fsBtn" class="btn">Fullscreen</button>
    <div id="fileName">—</div>
  </div>
  <div id="drop">Drop .ssbagpcm here</div>
  <div id="hint">Move mouse to feel parallax · Inpainted BG prevents stretching</div>

  <script type="module">
    // ES Module Three.js (plus de warning deprecation)
    import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

    const qs = new URLSearchParams(location.search);
    const remoteFile = qs.get("file") || "";

    const fileNameEl = document.getElementById("fileName");
    const drop = document.getElementById("drop");
    const openBtn = document.getElementById("openBtn");
    const fileInput = document.getElementById("fileInput");
    const fsBtn = document.getElementById("fsBtn");

    let renderer, scene, camera, mesh, uniforms;
    let targetDX=0, targetDY=0, curDX=0, curDY=0;

    let PARALLAX = 0.065;     // boosted x1.35 with manifest
    const GAMMA = 1.2;
    const DOF = true;
    const DOF_STRENGTH = 0.8;
    const DOF_FOCUS = 0.65;

    function initThree(){
      renderer = new THREE.WebGLRenderer({ antialias:false, alpha:false, powerPreference:"high-performance" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      document.body.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      camera = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
      const geom = new THREE.PlaneGeometry(2, 2);

      const frag = `
        precision highp float;
        varying vec2 vUV;
        uniform sampler2D uColor;
        uniform sampler2D uDepth;
        uniform sampler2D uBg;
        uniform vec2 uRes;
        uniform vec2 uDir;
        uniform float uParallax;
        uniform float uGamma;
        uniform bool uHasDepth;
        uniform bool uHasBg;
        uniform bool uEnableDOF;
        uniform float uDOFStrength;
        uniform float uDOFFocus;

        float readDepth(vec2 uv){
          vec3 rg = texture2D(uDepth, uv).rgb;
          float hi = floor(rg.r * 255.0 + 0.5);
          float lo = floor(rg.g * 255.0 + 0.5);
          return (hi * 256.0 + lo) / 65535.0; // far = 1
        }
        float nearF(vec2 uv){ float d = readDepth(uv); return pow(1.0 - d, uGamma); }
        vec4 sampleColor(vec2 uv){ return texture2D(uColor, clamp(uv, 0.0, 1.0)); }
        vec4 sampleBg(vec2 uv){ return texture2D(uBg, clamp(uv, 0.0, 1.0)); }

        void main(){
          vec2 uv = vUV;
          float minDim = min(uRes.x, uRes.y);
          vec2 pxUv = vec2(uParallax * (minDim / uRes.x), uParallax * (minDim / uRes.y));

          vec2 dir = uDir;
          float lenDir = max(length(dir), 1e-6);
          vec2 dirN = dir / lenDir;

          float nf = uHasDepth ? nearF(uv) : 0.0;
          vec2 offset = dir * pxUv * nf;
          vec2 uv0 = uv - offset;
          vec2 uv0c = clamp(uv0, 0.0, 1.0);

          vec4 col = sampleColor(uv0c);

          if (lenDir > 1e-4) {
            vec2 perp = normalize(vec2(-dirN.y, dirN.x));
            vec2 px = vec2(1.0/uRes.x, 1.0/uRes.y);
            vec4 c1 = sampleColor(uv0c + perp * 0.8 * px);
            vec4 c2 = sampleColor(uv0c - perp * 0.8 * px);
            col = mix(col, 0.5*(c1+c2), 0.12);
          }

          if (uHasDepth && uHasBg) {
            float clampOcc = step(uv0.x, 0.001) + step(uv0.y, 0.001) + step(0.999, uv0.x) + step(0.999, uv0.y);
            clampOcc = clamp(clampOcc, 0.0, 1.0);

            float nf0 = nf;
            float nf1 = nearF(uv0c);
            float occDepth = smoothstep(0.018, 0.12, max(0.0, nf0 - nf1));

            vec2 dpx = vec2(1.0/uRes.x, 0.0);
            vec2 dpy = vec2(0.0, 1.0/uRes.y);
            float dC = readDepth(uv);
            float gx = readDepth(clamp(uv + dpx, 0.0, 1.0)) - dC;
            float gy = readDepth(clamp(uv + dpy, 0.0, 1.0)) - dC;
            float g = max(abs(gx), abs(gy));
            float occG = smoothstep(0.007, 0.045, g);

            vec2 stepDir = dirN * vec2(1.0/uRes.x, 1.0/uRes.y);
            float d1 = readDepth(clamp(uv + stepDir * 1.5, 0.0, 1.0));
            float d2 = readDepth(clamp(uv + stepDir * 3.0, 0.0, 1.0));
            float occDir = smoothstep(0.006, 0.030, max(0.0, max(d1 - dC, d2 - dC)));

            // FIX: corrected parentheses on clamp (previously had an extra ')')
            float occ = clamp(max(clampOcc, max(occDepth, max(occG, occDir))), 0.0, 1.0);

            vec4 bg = sampleBg(uv0c) * 0.55
                    + sampleBg(uv0c + stepDir * 1.5) * 0.22
                    + sampleBg(uv0c + stepDir * 3.0) * 0.12
                    + sampleBg(uv0c + stepDir * 4.5) * 0.07
                    + sampleBg(uv0c + stepDir * 6.0) * 0.04;
            col = mix(col, bg, clamp(occ * 1.2, 0.0, 1.0));
          }

          if (uEnableDOF && uHasDepth && uDOFStrength > 0.0) {
            float z = readDepth(uv0c);
            float coc = abs(z - uDOFFocus) * uDOFStrength;
            coc = clamp(coc, 0.0, 3.2);
            if (coc > 0.01) {
              vec2 px = vec2(1.0/uRes.x, 1.0/uRes.y);
              vec4 acc = col; float wsum = 1.0;
              for (int i=0;i<12;i++){
                float a = 6.2831853 * float(i) / 12.0;
                vec2 o = vec2(cos(a), sin(a)) * coc;
                float w = 0.85;
                acc += sampleColor(uv0c + o*px) * w; wsum += w;
              }
              col = acc / wsum;
            }
          }

          gl_FragColor = col;
        }
      `;

      const vert = `
        varying vec2 vUV;
        void main(){ vUV = position.xy * 0.5 + 0.5; gl_Position = vec4(position, 1.0); }
      `;

      uniforms = {
        uColor:      { value: null },
        uDepth:      { value: null },
        uBg:         { value: null },
        uRes:        { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uDir:        { value: new THREE.Vector2(0,0) },
        uParallax:   { value: PARALLAX },
        uGamma:      { value: GAMMA },
        uHasDepth:   { value: false },
        uHasBg:      { value: false },
        uEnableDOF:  { value: DOF },
        uDOFStrength:{ value: DOF_STRENGTH },
        uDOFFocus:   { value: DOF_FOCUS }
      };

      const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
      mesh = new THREE.Mesh(geom, mat);
      scene.add(mesh);

      window.addEventListener("resize", onResize);
      window.addEventListener("mousemove", onMove);
      window.addEventListener("deviceorientation", onOrient);
      animate();
    }

    function onResize(){
      renderer.setSize(window.innerWidth, window.innerHeight);
      uniforms.uRes.value.set(window.innerWidth, window.innerHeight);
    }

    function onMove(e){
      const cx = window.innerWidth/2, cy = window.innerHeight/2;
      const nx = (e.clientX - cx)/(window.innerWidth/2);
      const ny = (e.clientY - cy)/(window.innerHeight/2);
      targetDX = -Math.max(-1, Math.min(1, nx));
      targetDY = -Math.max(-1, Math.min(1, ny));
    }
    function onOrient(e){
      if (e.beta === null || e.gamma === null) return;
      const nx = (e.gamma || 0) / 45;
      const ny = (e.beta || 0) / 45;
      targetDX = -Math.max(-1, Math.min(1, nx));
      targetDY = -Math.max(-1, Math.min(1, ny));
    }

    function animate(){
      const k = 0.82;
      curDX = curDX * k + targetDX * (1-k);
      curDY = curDY * k + targetDY * (1-k);
      uniforms.uDir.value.set(curDX, curDY);
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    function setFileName(n){ fileNameEl.textContent = n || "—"; }

    async function loadRemotePack(path){
      const mf = await fetch(`/api/manifest?file=${encodeURIComponent(path)}`).then(r=>r.json());
      if (typeof mf.parallax_default === "number") uniforms.uParallax.value = Math.max(0.0, mf.parallax_default) * 1.35;

      const colorTex = await loadTexture(mf.preview_url, true, true);
      let depthTex=null, bgTex=null;
      if (mf.depth_url) depthTex = await loadTexture(mf.depth_url, false, true, true);
      if (mf.bgfill_url) bgTex = await loadTexture(mf.bgfill_url, true, true);

      uniforms.uColor.value = colorTex;
      uniforms.uDepth.value = depthTex;
      uniforms.uBg.value = bgTex;
      uniforms.uHasDepth.value = !!depthTex;
      uniforms.uHasBg.value = !!bgTex;
      setFileName(mf.file || "(remote)");
    }

    async function loadLocalPack(file){
      const ab = await file.arrayBuffer();
      const zip = await JSZip.loadAsync(ab);
      const mf = JSON.parse(await zip.file("manifest.json").async("string"));
      if (typeof mf.parallax_default === "number") uniforms.uParallax.value = Math.max(0.0, mf.parallax_default) * 1.35;

      const blobPrev = await zip.file("preview.png").async("blob");
      const colorTex = await blobToTexture(blobPrev, true, true);
      let depthTex=null, bgTex=null;
      if (zip.file("depth.png")) {
        const blobDepth = await zip.file("depth.png").async("blob");
        depthTex = await blobToTexture(blobDepth, false, true, true);
      }
      if (zip.file("bgfill.png")) {
        const blobBg = await zip.file("bgfill.png").async("blob");
        bgTex = await blobToTexture(blobBg, true, true);
      }
      uniforms.uColor.value = colorTex;
      uniforms.uDepth.value = depthTex;
      uniforms.uBg.value = bgTex;
      uniforms.uHasDepth.value = !!depthTex;
      uniforms.uHasBg.value = !!bgTex;
      setFileName(file.name);
    }

    function loadImage(url){ return new Promise((res, rej)=>{ const i=new Image(); i.onload=()=>res(i); i.onerror=rej; i.crossOrigin="anonymous"; i.src=url; }); }
    function blobToImage(blob){ return new Promise((res)=>{ const u=URL.createObjectURL(blob); const i=new Image(); i.onload=()=>{URL.revokeObjectURL(u); res(i);} ; i.src=u; }); }

    async function loadTexture(url, sRGB=false, flipY=true, nearest=false){
      const img = await loadImage(url);
      const tex = new THREE.Texture(img);
      tex.needsUpdate = true;
      tex.flipY = flipY;
      if (sRGB) tex.colorSpace = THREE.SRGBColorSpace;
      if (nearest) { tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter; }
      return tex;
    }
    async function blobToTexture(blob, sRGB=false, flipY=true, nearest=false){
      const img = await blobToImage(blob);
      const tex = new THREE.Texture(img);
      tex.needsUpdate = true;
      tex.flipY = flipY;
      if (sRGB) tex.colorSpace = THREE.SRGBColorSpace;
      if (nearest) { tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter; }
      return tex;
    }

    openBtn.addEventListener("click", ()=> fileInput.click());
    fileInput.addEventListener("change", (e)=>{ if (e.target.files && e.target.files[0]) loadLocalPack(e.target.files[0]); });

    function prevent(e){ e.preventDefault(); e.stopPropagation(); }
    ["dragenter","dragover","dragleave","drop"].forEach(ev => window.addEventListener(ev, prevent));
    window.addEventListener("dragenter", ()=>{ drop.style.display="flex"; });
    window.addEventListener("dragleave", ()=>{ drop.style.display="none"; });
    window.addEventListener("drop", (e)=>{
      drop.style.display="none";
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f && f.name.toLowerCase().endsWith(".ssbagpcm")) loadLocalPack(f);
    });

    fsBtn.addEventListener("click", ()=>{
      const el = document.documentElement;
      if (!document.fullscreenElement) el.requestFullscreen?.(); else document.exitFullscreen?.();
    });

    initThree();
    if (remoteFile) loadRemotePack(remoteFile);
  </script>
  <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
</body>
</html>
"""

# =============================================================================
# CLI entry point — robust, minimal, friendly
# =============================================================================
def main():
    """
    CLI orchestrator:
      - When given an input image path: build a .ssbagpcm next to it (or at --output path).
      - When --runserver: spin up the viewer on localhost and optionally open the generated pack.
      - When nothing is provided: print usage examples.
    """
    parser = argparse.ArgumentParser(
        description="Single-file SSBAGPCM — 2.5D packaging (DAV2 + MiDaS + LaMa) + richly documented viewer"
    )
    parser.add_argument("input", type=str, nargs="?", help="input image (jpg/jpeg/png/webp/avif). If omitted, no generation.")
    parser.add_argument("--output", type=str, default="", help="output .ssbagpcm path (defaults next to input)")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS, help="number of layers (soft RBF near-biased)")
    parser.add_argument("--blur", type=float, default=DEFAULT_BLUR_SIGMA, help="layer alpha feather sigma")
    parser.add_argument("--parallax", type=float, default=DEFAULT_PARALLAX, help="default parallax amplitude (viewer boosts x1.35)")
    parser.add_argument("--runserver", action="store_true", help="launch minimal viewer server (http://127.0.0.1:8765)")
    parser.add_argument("--port", type=int, default=8765, help="viewer port (default 8765)")
    args = parser.parse_args()

    out_path = None
    if args.input:
        in_path = args.input
        if not os.path.isfile(in_path):
            print(f"File not found: {in_path}")
            return
        # Where to write the pack (.ssbagpcm)
        out_path = args.output or os.path.splitext(os.path.abspath(in_path))[0] + ".ssbagpcm"
        # Build the pack (blocking call, logs to console)
        info = process_image_to_ssbagpcm(
            in_path, out_path,
            layers=args.layers,
            blur_sigma=args.blur,
            parallax=args.parallax
        )
        print("Generated:", info)

    if args.runserver:
        # If we just generated a pack, craft a URL that auto-loads it via ?file=
        if out_path:
            url = f"http://127.0.0.1:{args.port}/?file={quote(os.path.abspath(out_path))}"
        else:
            url = f"http://127.0.0.1:{args.port}/"
        print("Viewer →", url)
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
        start_server(args.port)
    else:
        if not args.input:
            print("No generation, no server. Examples:")
            print("  - Generate:          python server.py image.jpg")
            print("  - Start viewer:      python server.py --runserver")
            print("  - Generate + viewer: python server.py image.jpg --runserver")

# Standard Python entry point guard
if __name__ == "__main__":
    main()