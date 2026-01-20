"""Utilities for ComfyUI integration: image format conversion and rendering setup."""

import os
import ctypes
import numpy as np
import torch
from typing import List, Tuple
from numpy.typing import NDArray


def setup_headless_rendering():
    """Configure OpenGL for headless rendering.

    ComfyUI may run without a display. Sets PYOPENGL_PLATFORM to EGL
    (GPU-accelerated) if available, otherwise OSMesa (software rendering).

    Must be called before any pyrender/body2colmap imports.
    """
    if "PYOPENGL_PLATFORM" in os.environ:
        # Already configured
        print(f"[Body2COLMAP] Using pre-configured platform: {os.environ['PYOPENGL_PLATFORM']}")
        return

    # Try EGL first (GPU-accelerated on NVIDIA), fallback to OSMesa
    # The actual test happens when body2colmap creates its first renderer
    try:
        # Check if EGL libraries are available
        ctypes.CDLL("libEGL.so.1")
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[Body2COLMAP] Configured for EGL rendering (GPU-accelerated)")
    except (OSError, FileNotFoundError):
        # EGL not available, use OSMesa
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        print("[Body2COLMAP] Configured for OSMesa rendering (software, slower)")


def rendered_to_comfy(images: List[NDArray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert list of rendered RGBA images to ComfyUI IMAGE and MASK formats.

    Args:
        images: List of [H, W, 4] uint8 arrays in [0, 255] (RGBA)

    Returns:
        Tuple of:
        - images: Tensor of shape [B, H, W, 3] in [0, 1] float32 (RGB)
        - masks: Tensor of shape [B, H, W] in [0, 1] float32 (alpha channel)
    """
    if not images:
        raise ValueError("Empty image list")

    # Stack into batch
    batch = np.stack(images, axis=0)  # [B, H, W, 4]

    # Split RGB and Alpha
    rgb = batch[..., :3]  # [B, H, W, 3]
    alpha = batch[..., 3]  # [B, H, W]

    # Convert to float [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    alpha = alpha.astype(np.float32) / 255.0

    # Invert alpha for ComfyUI MASK convention
    # ComfyUI: 1.0 = visible/keep, 0.0 = masked/hidden
    # Renderer: alpha 1.0 = opaque content, 0.0 = transparent background
    # We want mask to be 1.0 where there's NO content (background)
    mask = 1.0 - alpha

    # Convert to torch tensors
    return torch.from_numpy(rgb), torch.from_numpy(mask)


def comfy_to_cv2(images: torch.Tensor) -> List[NDArray]:
    """
    Convert ComfyUI IMAGE to list of OpenCV BGR images for saving.

    Args:
        images: Tensor of shape [B, H, W, 3] in [0, 1] (RGB)

    Returns:
        List of [H, W, 3] uint8 BGR arrays
    """
    # To numpy
    batch = images.cpu().numpy()  # [B, H, W, 3]

    # Scale to [0, 255]
    batch = (batch * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    batch = batch[..., ::-1]

    return [batch[i] for i in range(batch.shape[0])]


def comfy_to_rgb(images: torch.Tensor) -> List[NDArray]:
    """
    Convert ComfyUI IMAGE to list of RGB numpy arrays.

    Args:
        images: Tensor of shape [B, H, W, 3] in [0, 1] (RGB)

    Returns:
        List of [H, W, 3] uint8 RGB arrays
    """
    # To numpy
    batch = images.cpu().numpy()  # [B, H, W, 3]

    # Scale to [0, 255]
    batch = (batch * 255).astype(np.uint8)

    return [batch[i] for i in range(batch.shape[0])]
