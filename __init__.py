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

from .core.comfy_utils import setup_headless_rendering

# Configure headless rendering before importing nodes
setup_headless_rendering()

from .nodes.path_nodes import (
    Body2COLMAP_CircularPath,
    Body2COLMAP_SinusoidalPath,
    Body2COLMAP_HelicalPath,
)
from .nodes.render_node import Body2COLMAP_Render
from .nodes.export_node import Body2COLMAP_ExportCOLMAP

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Body2COLMAP_CircularPath": Body2COLMAP_CircularPath,
    "Body2COLMAP_SinusoidalPath": Body2COLMAP_SinusoidalPath,
    "Body2COLMAP_HelicalPath": Body2COLMAP_HelicalPath,
    "Body2COLMAP_Render": Body2COLMAP_Render,
    "Body2COLMAP_ExportCOLMAP": Body2COLMAP_ExportCOLMAP,
}

# Display names for ComfyUI node menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "Body2COLMAP_CircularPath": "🌐 Circular Path",
    "Body2COLMAP_SinusoidalPath": "🌊 Sinusoidal Path",
    "Body2COLMAP_HelicalPath": "🌀 Helical Path",
    "Body2COLMAP_Render": "🎬 Render Multi-View",
    "Body2COLMAP_ExportCOLMAP": "📦 Export COLMAP",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Debug: confirm nodes are loaded
print(f"[Body2COLMAP] Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
    print(f"  - {node_name}")
