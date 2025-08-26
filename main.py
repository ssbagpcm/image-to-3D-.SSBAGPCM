#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Cross-platform (Windows/macOS/Linux) “install-everything” bootstrapper for the SSBAGPCM project.

What this script does:
- Installs all Python dependencies (FastAPI, Uvicorn, NumPy, Pillow, OpenCV contrib, HuggingFace Hub, TIMM, JS deps not needed since loaded via CDN).
- Installs PyTorch in a cross-platform way:
    * Default: CPU build (safe everywhere) from the official PyTorch wheel index.
    * Auto-detect NVIDIA GPU on Windows/Linux -> propose CUDA 12.1 build (--torch-build to override).
    * macOS: CPU build also enables MPS (Apple Metal) inside the same wheel family.
- Installs Depth Anything V2 (from GitHub). If pip git install fails (or git missing),
  the script prints a clear fallback message (MiDaS will still work).
- Pre-downloads key model weights:
    * Depth Anything V2 weights via HuggingFace Hub (vitl by default),
    * MiDaS model/transforms via torch.hub (caches in ~/.cache/torch/hub),
    * LaMa weights by instantiating simple-lama-inpainting and doing a tiny inpaint.
- Verifies imports and environment.

After running:
- You can generate a pack (.ssbagpcm):     python server.py your_image.jpg
- You can start the viewer:                python server.py --runserver
- Or do both:                              python server.py your_image.jpg --runserver

Important notes:
- This script installs into the CURRENT Python environment (venv recommended).
- It uses pip via `sys.executable -m pip`. No system package managers are touched.
- If you want to force CPU/GPU builds for torch, use --torch-build cpu|cu121|cu118.
- Use --non-interactive to avoid any prompt (defaults to CPU if unsure).
"""

import sys
import os
import platform
import subprocess
import shutil
import site
import json
import time
import pathlib
from typing import List, Optional

# ---------------------------
# Tunable constants / defaults
# ---------------------------
DEP_DIR = pathlib.Path("./deps").absolute()  # where we clone vendored repos if needed (e.g., Depth-Anything-V2)
DEP_DIR.mkdir(parents=True, exist_ok=True)

# Depth Anything V2 Git repo
DAV2_GIT = "https://github.com/DepthAnything/Depth-Anything-V2.git"

# Torch official wheel indexes
TORCH_IDX_CPU   = "https://download.pytorch.org/whl/cpu"
TORCH_IDX_CU118 = "https://download.pytorch.org/whl/cu118"
TORCH_IDX_CU121 = "https://download.pytorch.org/whl/cu121"

# Base pip packages (versions not pinned here to keep compatibility wide)
BASE_PACKAGES = [
    # Web server
    "fastapi",
    "uvicorn",

    # Image / numeric
    "numpy",
    "pillow",

    # OpenCV contrib (ximgproc guidedFilter/jointBilateral used in project)
    "opencv-contrib-python",

    # Model hub + utilities
    "huggingface_hub",
    "timm",

    # Inpainting wrapper (auto-downloads LaMa)
    "simple-lama-inpainting",

    # Reliability helpers
    "requests",
    "packaging",
]

# Try to install Depth Anything V2 via pip git+ URL
DAV2_PIP_SPEC = f"git+{DAV2_GIT}"

# ---------------------------
# Utility helpers
# ---------------------------
def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a subprocess command with stdout/stderr inherited, raise on failure by default.
    """
    print(f"[run] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)

def pip_install(pkgs: List[str], extra_args: Optional[List[str]] = None) -> None:
    """
    Install pip packages using the current Python interpreter.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if extra_args:
        cmd += extra_args
    cmd += pkgs
    run(cmd)

def pip_upgrade_pip() -> None:
    """
    Upgrade pip itself (often fixes wheel resolution issues).
    """
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def has_command(name: str) -> bool:
    """
    Check if a command exists in PATH (cross-platform).
    """
    return shutil.which(name) is not None

def detect_os() -> str:
    """
    Return 'Windows', 'Darwin' (macOS), or 'Linux'.
    """
    return platform.system()

def detect_nvidia_gpu() -> bool:
    """
    Rudimentary NVIDIA GPU detection: presence of nvidia-smi in PATH is a good proxy.
    """
    return has_command("nvidia-smi")

def ask_user_choice(prompt: str, choices: List[str], default: str) -> str:
    """
    Interactive choice prompt; returns the chosen string (lowercased).
    Falls back to default if empty input.
    """
    txt = f"{prompt} [{'/'.join(choices)}] (default: {default}): "
    ans = input(txt).strip().lower()
    if not ans:
        return default
    if ans not in choices:
        print(f"Invalid choice '{ans}', defaulting to '{default}'.")
        return default
    return ans

# ---------------------------
# Torch installer
# ---------------------------
def install_torch(build: str, non_interactive: bool = False) -> None:
    """
    Install PyTorch, torchvision, torchaudio using the official wheel indexes.

    build:
      - "cpu"   -> TORCH_IDX_CPU
      - "cu118" -> TORCH_IDX_CU118
      - "cu121" -> TORCH_IDX_CU121

    On macOS: CPU index is fine; it includes MPS (Apple Metal) support in modern torch wheels.
    """
    idx = TORCH_IDX_CPU
    if build == "cu118":
        idx = TORCH_IDX_CU118
    elif build == "cu121":
        idx = TORCH_IDX_CU121

    print(f"[torch] Installing PyTorch build '{build}' from index: {idx}")
    pip_install(["torch", "torchvision", "torchaudio"], extra_args=["--index-url", idx])

    # Quick sanity check
    try:
        import torch  # noqa
        print(f"[torch] Installed torch version: {torch.__version__}")
        if build.startswith("cu"):
            # Confirm CUDA is available
            import torch as _t
            if not _t.cuda.is_available():
                print("[torch] WARNING: CUDA build installed but torch.cuda.is_available() == False.")
                print("         Make sure you have a compatible NVIDIA driver and CUDA runtime.")
    except Exception as e:
        print(f"[torch] ERROR: import torch failed: {e}")
        if not non_interactive:
            print("Press Enter to continue anyway (or Ctrl+C to abort)...")
            try:
                input()
            except KeyboardInterrupt:
                sys.exit(1)

# ---------------------------
# Depth Anything V2 installer
# ---------------------------
def install_depth_anything_v2(non_interactive: bool = False) -> bool:
    """
    Try to install the Depth Anything V2 Python module so we can import:
        from depth_anything_v2.dpt import DepthAnythingV2

    Strategy:
      1) pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
      2) If that fails and git exists: clone into ./deps and try to pip install from there.
      3) If still failing: we fallback (MiDaS will still work). Return False.
    """
    print("[DAV2] Installing Depth Anything V2 (from GitHub)...")
    try:
        pip_install([DAV2_PIP_SPEC])
        import importlib
        importlib.import_module("depth_anything_v2.dpt")
        print("[DAV2] OK — depth_anything_v2 import works.")
        return True
    except Exception as e:
        print(f"[DAV2] First attempt failed: {e}")

    if not has_command("git"):
        print("[DAV2] git is not available on your system; cannot clone the repo.")
        print("[DAV2] Fallback to MiDaS only (depth still works, slightly less detailed).")
        return False

    try:
        # Clone into deps directory
        repo_dir = DEP_DIR / "Depth-Anything-V2"
        if repo_dir.exists():
            print(f"[DAV2] Repo already cloned at {repo_dir}")
        else:
            print(f"[DAV2] Cloning into {repo_dir} ...")
            run(["git", "clone", "--depth", "1", DAV2_GIT, str(repo_dir)])

        print("[DAV2] Installing from local clone via pip...")
        pip_install([str(repo_dir)])

        import importlib
        importlib.import_module("depth_anything_v2.dpt")
        print("[DAV2] OK — depth_anything_v2 import works (local clone).")
        return True
    except Exception as e:
        print(f"[DAV2] Failed to install from local clone: {e}")
        print("[DAV2] Fallback to MiDaS only.")
        return False

# ---------------------------
# Model pre-download / warm-up
# ---------------------------
def predownload_dav2_weights() -> None:
    """
    Use HuggingFace Hub API to pre-fetch the large encoder weights for Depth Anything V2.
    This avoids first-run latency at generation time.
    """
    print("[DAV2] Pre-downloading weights via HuggingFace Hub (Large encoder)...")
    try:
        from huggingface_hub import hf_hub_download
        repo = "depth-anything/Depth-Anything-V2-Large"
        fname = "depth_anything_v2_vitl.pth"
        path = hf_hub_download(repo_id=repo, filename=fname, repo_type="model")
        print(f"[DAV2] Weights cached at: {path}")
    except Exception as e:
        print(f"[DAV2] WARNING: Could not pre-download weights: {e} (will download on first use).")

def predownload_midas() -> None:
    """
    Trigger MiDaS model + transform download via torch.hub, then release.
    """
    print("[MiDaS] Pre-downloading model + transforms via torch.hub ...")
    try:
        import torch
        try:
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        except TypeError:
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model.eval()
        try:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        except TypeError:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _ = transforms.dpt_transform
        print("[MiDaS] Cached successfully.")
    except Exception as e:
        print(f"[MiDaS] WARNING: Could not pre-download model/transforms: {e}")

def predownload_lama() -> None:
    """
    Instantiate simple-lama-inpainting and run a tiny inpaint to trigger weight download.
    """
    print("[LaMa] Pre-downloading weights (simple-lama-inpainting) ...")
    try:
        from simple_lama_inpainting import SimpleLama
        import numpy as np
        import cv2

        lama = SimpleLama()
        # Tiny dummy image/mask (64x64) just to trigger model load
        img = np.zeros((64,64,3), dtype=np.uint8)
        img[:] = (127,127,127)
        mask = np.zeros((64,64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        _ = lama(img, mask)
        print("[LaMa] Cached successfully.")
    except Exception as e:
        print(f"[LaMa] WARNING: Could not pre-download weights: {e}")

# ---------------------------
# Full installer
# ---------------------------
def install_all(torch_build: str = "auto", non_interactive: bool = False) -> None:
    """
    End-to-end installation for all Python deps + models.
    torch_build:
      - "auto": detect NVIDIA GPU on Win/Linux -> attempt CUDA 12.1, else CPU
      - "cpu" | "cu118" | "cu121": force a specific build
    non_interactive:
      - True: no prompt, default choices applied automatically.
    """
    os_name = detect_os()
    print(f"[env] OS detected: {os_name}")
    print(f"[env] Python: {sys.version.split()[0]} @ {sys.executable}")

    # Always upgrade pip to latest (reduces wheel resolution headaches)
    pip_upgrade_pip()

    # Decide torch build
    chosen_build = "cpu"
    if torch_build == "auto":
        if os_name in ("Windows", "Linux") and detect_nvidia_gpu():
            if non_interactive:
                chosen_build = "cu121"
            else:
                # Offer user choice
                print("[torch] NVIDIA GPU detected via nvidia-smi.")
                chosen_build = ask_user_choice("Install torch build", ["cu121", "cpu"], "cu121")
        else:
            chosen_build = "cpu"
    else:
        if torch_build not in ("cpu", "cu118", "cu121"):
            print(f"[torch] Unknown build '{torch_build}', defaulting to CPU.")
            chosen_build = "cpu"
        else:
            chosen_build = torch_build

    # Install torch + friends
    install_torch(chosen_build, non_interactive=non_interactive)

    # Install base packages
    print("[pip] Installing base packages ...")
    pip_install(BASE_PACKAGES)

    # Depth Anything V2 (module)
    dav2_ok = install_depth_anything_v2(non_interactive=non_interactive)
    if dav2_ok:
        predownload_dav2_weights()
    else:
        print("[DAV2] Skipping weight pre-download since module isn't available.")

    # Preload MiDaS + LaMa
    predownload_midas()
    predownload_lama()

    # Final sanity checks
    print("[check] Verifying core imports ...")
    try:
        import numpy as _np  # noqa
        import PIL          # noqa
        import cv2          # noqa
        import fastapi      # noqa
        import uvicorn      # noqa
        import huggingface_hub  # noqa
        import timm         # noqa
        import simple_lama_inpainting  # noqa
        import torch as _t
        print(f"[check] torch={_t.__version__}, cuda_available={_t.cuda.is_available()}")
        # Depth Anything V2 optional
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # noqa
            print("[check] depth_anything_v2 import OK")
        except Exception as e:
            print(f"[check] depth_anything_v2 import FAILED (pipeline will fallback to MiDaS): {e}")
    except Exception as e:
        print(f"[check] ERROR: import check failed: {e}")

    print("\n===== Setup completed =====")
    print("Next steps:")
    print("  - Generate a .ssbagpcm from an image:")
    print("      python server.py your_image.jpg")
    print("  - Start the viewer:")
    print("      python server.py --runserver")
    print("  - Generate + open viewer:")
    print("      python server.py your_image.jpg --runserver")
    print("")
    print("If you installed a CUDA build of PyTorch, ensure your NVIDIA drivers are up to date.")
    print("If Depth Anything V2 isn't available, the pipeline still works using MiDaS fallback.")

# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="All-in-one installer for SSBAGPCM (cross-platform).")
    p.add_argument("--torch-build", type=str, default="auto",
                   help="Torch build to install: auto|cpu|cu118|cu121 (default: auto)")
    p.add_argument("--non-interactive", action="store_true",
                   help="Run without prompts (defaults to CPU unless NVIDIA GPU detected -> cu121)")
    args = p.parse_args()

    try:
        install_all(torch_build=args.torch_build, non_interactive=args.non_interactive)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()