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
        initial_rotation: Degrees offset applied after auto-orient [optional, from mesh renderer]
    """
    cameras: List[Any]  # List[Camera] - avoiding import here
    image_names: List[str]
    points_3d: Tuple[NDArray[np.float32], NDArray[np.uint8]]
    resolution: Tuple[int, int]
    splat_path: Optional[str]  # Optional field for splat integration
    framing_bounds: Optional[Dict[str, Tuple[NDArray[np.float32], NDArray[np.float32]]]]  # preset -> (min_corner, max_corner)
    initial_rotation: Optional[float]  # Degrees offset after auto-orient (for splat renderer reuse)


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


class B2C_IMAGE_WARP(TypedDict, total=False):
    """Warp data for transforming a reference image to match a rendered view.

    Produced by the Render node when ``override_cam_from_mesh`` is enabled.
    Consumed by the Generate FirstLast node to warp the reference photo so
    it aligns with the skeleton rendered at the anchor frame — frame 0 for a
    circular path, solved for on a helical one and reported as
    ``anchor_frame_index`` in b2c_data.

    Attributes:
        camera: Camera object for the anchor frame (framed intrinsics +
            look_at rotation).
        original_focal_length: SAM-3D-Body focal length in pixels (for the
            original photo resolution — not render_size).
        render_size: (width, height) of the rendered output.
        bg_color: Mesh background color as (r, g, b) floats in [0, 1].
    """
    camera: Any  # body2colmap Camera object  [required]
    original_focal_length: float  # [required]
    render_size: Tuple[int, int]  # [required]
    bg_color: Tuple[float, float, float]  # [optional]


# Custom type identifier for Gaussian Splat scenes
# The actual data is a SplatScene object from body2colmap.splat_scene
SPLAT_SCENE = "SPLAT_SCENE"
