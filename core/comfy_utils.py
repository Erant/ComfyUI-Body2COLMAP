"""Utilities for ComfyUI integration: image format conversion and rendering setup."""

import os
import numpy as np
import torch
from typing import List, Tuple
from numpy.typing import NDArray


def setup_headless_rendering():
    """Configure OpenGL for headless rendering.

    ComfyUI may run without a display. Try EGL first (GPU-accelerated),
    fall back to OSMesa (software rendering) if EGL is unavailable.
    """
    if "PYOPENGL_PLATFORM" in os.environ:
        # Already configured
        return

    try:
        # Try EGL first (NVIDIA GPUs, faster)
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        # Quick test to see if EGL works
        r = pyrender.OffscreenRenderer(64, 64)
        r.delete()
        print("[Body2COLMAP] Using EGL for rendering")
    except Exception as e:
        # Fall back to OSMesa (software rendering, slower but more compatible)
        try:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender
            r = pyrender.OffscreenRenderer(64, 64)
            r.delete()
            print("[Body2COLMAP] Using OSMesa for rendering")
        except Exception as e2:
            # If both fail, just continue - renderer will be created on first use
            print(f"[Body2COLMAP] Warning: Could not initialize renderer ({e2})")


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
