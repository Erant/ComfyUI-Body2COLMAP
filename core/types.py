"""Custom data types for Body2COLMAP ComfyUI nodes."""

from typing import TypedDict, Any, Dict


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


# Custom type identifier for Gaussian Splat scenes
# The actual data is a SplatScene object from body2colmap.splat_scene
SPLAT_SCENE = "SPLAT_SCENE"
