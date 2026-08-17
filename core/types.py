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
        focal_length_mm: 35mm-equivalent focal length, 0 = auto [optional]
        orbit_target: Orbit center / look-at point, shape (3,) [optional]
        forward_azimuth_deg: Orbit azimuth that faces the front of the subject
                       [optional, used by Filter FoV and Rotate Views]
        anchor_frame_index: Frame that sits at the original SAM-3D-Body camera
                       [optional, written when override_cam_from_mesh is on].
                       Informational only — it goes stale as soon as Drop Views
                       / Rotate Views / Filter FoV reorder or subset the views.
        anchor_position: World position of that camera, shape (3,) [optional].
                       This is the durable anchor key: Inject Anchor matches
                       frames against it by position, and it survives a
                       Save → Load round-trip (as a plain list, so coerce with
                       np.asarray before doing arithmetic on it).

    Note: keys other than cameras/image_names/points_3d/resolution/splat_path
    round-trip through Save Dataset's ``b2c_extras`` only if they are
    JSON-serializable (numpy arrays are converted via ``.tolist()``), so never
    stash objects like Camera here.
    """
    cameras: List[Any]  # List[Camera] - avoiding import here
    image_names: List[str]
    points_3d: Tuple[NDArray[np.float32], NDArray[np.uint8]]
    resolution: Tuple[int, int]
    splat_path: Optional[str]  # Optional field for splat integration
    framing_bounds: Optional[Dict[str, Tuple[NDArray[np.float32], NDArray[np.float32]]]]  # preset -> (min_corner, max_corner)
    initial_rotation: Optional[float]  # Degrees offset after auto-orient (for splat renderer reuse)
    focal_length_mm: Optional[float]  # 0 = auto, >0 = explicit 35mm equivalent
    orbit_target: Optional[NDArray[np.float32]]  # (3,) orbit center
    forward_azimuth_deg: Optional[float]  # Orbit azimuth that = subject front
    anchor_frame_index: Optional[int]  # Frame at the original camera (goes stale on reorder)
    anchor_position: Optional[NDArray[np.float32]]  # (3,) position of that camera


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
    circular orbit, a solved-for index for a helical one (see
    ``anchor_frame_index`` in B2C_COLMAP_METADATA).

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
