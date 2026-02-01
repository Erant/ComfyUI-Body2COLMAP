"""Custom data types for Body2COLMAP ComfyUI nodes."""

from typing import TypedDict, Any, Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


class B2C_PATH_CONFIG(TypedDict):
    """Path configuration passed from path generator to render node.

    This is pure configuration - no cameras or computed data.
    The render node uses this to generate the actual camera path.

    Attributes:
        pattern: Path pattern type ("circular", "sinusoidal", "helical")
        params: Pattern-specific parameters dict
    """
    pattern: str
    params: Dict[str, Any]


class B2C_COLMAP_METADATA(TypedDict):
    """COLMAP metadata for dataset serialization.

    Contains all data needed to write COLMAP format files.
    Used by Generate Metadata, Save COLMAP, and Load COLMAP nodes.

    Attributes:
        cameras: List of Camera objects (from body2colmap)
        image_names: Standardized filenames (frame_00001_.png, frame_00002_.png, ...)
        points_3d: Tuple of (positions, colors) arrays for initial point cloud
        resolution: Image resolution (width, height)
    """
    cameras: List[Any]  # List[Camera] - avoiding import here
    image_names: List[str]
    points_3d: Tuple[NDArray[np.float32], NDArray[np.uint8]]
    resolution: Tuple[int, int]


# Custom type identifier for Gaussian Splat scenes
# The actual data is a SplatScene object from body2colmap.splat_scene
SPLAT_SCENE = "SPLAT_SCENE"
