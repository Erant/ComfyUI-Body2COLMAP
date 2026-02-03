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
        framing: Camera framing preset ("full", "torso", "bust", "head")
    """
    pattern: str
    params: Dict[str, Any]
    framing: str


class B2C_COLMAP_METADATA(TypedDict, total=False):
    """COLMAP metadata for dataset serialization.

    Contains all data needed to write COLMAP format files.
    Used by Generate Metadata, Save COLMAP, and Load COLMAP nodes.

    Attributes:
        cameras: List of Camera objects (from body2colmap) [required]
        image_names: Standardized filenames (frame_00001_.png, frame_00002_.png, ...) [required]
        points_3d: Tuple of (positions, colors) arrays for initial point cloud [required]
        resolution: Image resolution (width, height) [required]
        splat_path: Path to trained Gaussian splat PLY file [optional, None or "" if no splat]
        framing_bounds: Dict mapping framing presets to their bounding boxes [optional]
                       e.g., {"full": (min, max), "torso": (min, max), "bust": (min, max), "head": (min, max)}
    """
    cameras: List[Any]  # List[Camera] - avoiding import here
    image_names: List[str]
    points_3d: Tuple[NDArray[np.float32], NDArray[np.uint8]]
    resolution: Tuple[int, int]
    splat_path: Optional[str]  # Optional field for splat integration
    framing_bounds: Optional[Dict[str, Tuple[NDArray[np.float32], NDArray[np.float32]]]]  # preset -> (min_corner, max_corner)


# Custom type identifier for Gaussian Splat scenes
# The actual data is a SplatScene object from body2colmap.splat_scene
SPLAT_SCENE = "SPLAT_SCENE"
