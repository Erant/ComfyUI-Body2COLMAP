"""Custom data types for Body2COLMAP ComfyUI nodes."""

from typing import TypedDict, Any, Dict, List, Tuple, Optional, Union
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


class B2C_FACE_LANDMARKS(TypedDict):
    """Face landmark detection results for body2colmap ingestion.

    Mirrors the JSON contract used by body2colmap's FaceLandmarkIngest:
    the ``source`` field selects which ``from_*`` converter to call, and
    the remaining fields provide the data that converter needs.

    Currently only "mediapipe" is supported.  Future sources (e.g. "dlib",
    "insightface") would add their own ``from_*`` methods to
    FaceLandmarkIngest and use the same dispatching pattern.

    Attributes:
        source: Identifier for the landmark format (e.g. "mediapipe").
            Used to dispatch to the correct FaceLandmarkIngest converter.
        landmarks: Raw landmark coordinates, shape (N, 3) float32.
            For mediapipe: N is 478 (refined) or 468, coords are
            normalized to [0,1] relative to image dimensions.
        image_size: (width, height) of the source image in pixels.
            Required for correct denormalization of coordinates.
    """
    source: str
    landmarks: NDArray[np.float32]
    image_size: Tuple[int, int]


# Custom type identifier for Gaussian Splat scenes
# The actual data is a SplatScene object from body2colmap.splat_scene
SPLAT_SCENE = "SPLAT_SCENE"
