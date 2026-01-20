"""Path generation utilities wrapping body2colmap library functionality."""

from typing import Tuple
from body2colmap.scene import Scene
from body2colmap.utils import compute_auto_orbit_radius


def compute_smart_orbit_radius(
    scene: Scene,
    render_size: Tuple[int, int],
    focal_length: float,
    fill_ratio: float = 0.8
) -> float:
    """
    Compute orbit radius using smart algorithm from body2colmap library.

    This is a thin wrapper around body2colmap.utils.compute_auto_orbit_radius
    that accepts a Scene object for convenience.

    Args:
        scene: Scene to frame
        render_size: (width, height) in pixels
        focal_length: Focal length in pixels
        fill_ratio: How much of viewport to fill (0.0 to 1.0)

    Returns:
        Orbit radius in world units
    """
    return compute_auto_orbit_radius(
        bounds=scene.get_bounds(),
        render_size=render_size,
        focal_length=focal_length,
        fill_ratio=fill_ratio
    )
