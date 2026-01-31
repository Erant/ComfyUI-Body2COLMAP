"""ComfyUI-Body2COLMAP: Generate multi-view training data for 3D Gaussian Splatting.

This node pack integrates body2colmap with ComfyUI-SAM3DBody to generate
multi-view rendered images and COLMAP camera parameters from SAM-3D-Body
mesh reconstructions.

Example workflow:
    Load Image → SAM3DBodyProcess → HelicalPath → Render → SaveImage
                                                         ↓
                                                   ExportCOLMAP

For more information, see:
- https://github.com/Erant/ComfyUI-Body2COLMAP
"""

# Configure headless rendering for Linux BEFORE importing body2colmap/pyrender.
# This must happen before any pyrender imports as it checks PYOPENGL_PLATFORM at import time.
import os
import sys

# Diagnostic: Check if OpenGL/pyrender was already imported (would be too late to configure)
_opengl_already_imported = 'OpenGL' in sys.modules
_pyrender_already_imported = 'pyrender' in sys.modules

print(f"[Body2COLMAP] Platform: {sys.platform}")
print(f"[Body2COLMAP] DISPLAY env: {os.environ.get('DISPLAY', '<not set>')}")
print(f"[Body2COLMAP] PYOPENGL_PLATFORM env: {os.environ.get('PYOPENGL_PLATFORM', '<not set>')}")
print(f"[Body2COLMAP] OpenGL already imported: {_opengl_already_imported}")
print(f"[Body2COLMAP] pyrender already imported: {_pyrender_already_imported}")

if sys.platform.startswith('linux'):
    # On Linux, always prefer EGL for offscreen rendering - it works reliably both
    # with and without a display, and avoids hangs when DISPLAY is set but X server
    # is not accessible.
    if 'PYOPENGL_PLATFORM' not in os.environ:
        # Try EGL first (GPU-accelerated), fall back to OSMesa (software)
        try:
            import ctypes
            ctypes.CDLL('libEGL.so.1')
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            print("[Body2COLMAP] Set PYOPENGL_PLATFORM=egl (EGL available)")
        except (OSError, FileNotFoundError):
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            print("[Body2COLMAP] Set PYOPENGL_PLATFORM=osmesa (EGL not available)")
    else:
        print(f"[Body2COLMAP] PYOPENGL_PLATFORM already set, not modifying")
else:
    print(f"[Body2COLMAP] OpenGL platform setup skipped (not Linux)")

from .nodes.path_nodes import (
    Body2COLMAP_CircularPath,
    Body2COLMAP_SinusoidalPath,
    Body2COLMAP_HelicalPath,
)
from .nodes.render_node import Body2COLMAP_Render
from .nodes.splat_loader_node import Body2COLMAP_LoadSplat
from .nodes.splat_render_node import Body2COLMAP_RenderSplat
from .nodes.export_node import Body2COLMAP_ExportCOLMAP

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Body2COLMAP_CircularPath": Body2COLMAP_CircularPath,
    "Body2COLMAP_SinusoidalPath": Body2COLMAP_SinusoidalPath,
    "Body2COLMAP_HelicalPath": Body2COLMAP_HelicalPath,
    "Body2COLMAP_Render": Body2COLMAP_Render,
    "Body2COLMAP_LoadSplat": Body2COLMAP_LoadSplat,
    "Body2COLMAP_RenderSplat": Body2COLMAP_RenderSplat,
    "Body2COLMAP_ExportCOLMAP": Body2COLMAP_ExportCOLMAP,
}

# Display names for ComfyUI node menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "Body2COLMAP_CircularPath": "🌐 Circular Path",
    "Body2COLMAP_SinusoidalPath": "🌊 Sinusoidal Path",
    "Body2COLMAP_HelicalPath": "🌀 Helical Path",
    "Body2COLMAP_Render": "🎬 Render Multi-View (Mesh)",
    "Body2COLMAP_LoadSplat": "✨ Load Gaussian Splat",
    "Body2COLMAP_RenderSplat": "🎬 Render Multi-View (Splat)",
    "Body2COLMAP_ExportCOLMAP": "📦 Export COLMAP",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Debug: confirm nodes are loaded
print(f"[Body2COLMAP] Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
    print(f"  - {node_name}")
