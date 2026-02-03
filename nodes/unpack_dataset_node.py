"""Dataset Unpack node - exposes internal metadata from B2C_COLMAP_METADATA."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Body2COLMAP_UnpackDataset:
    """Unpack Body2COLMAP dataset metadata into individual outputs."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "unpack"
    RETURN_TYPES = ("SPLAT_SCENE", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("splat_scene", "splat_path", "width", "height", "camera_count", "point_count")
    OUTPUT_TOOLTIPS = (
        "Gaussian Splat scene (None if not present)",
        "Path to splat PLY file (empty string if not present)",
        "Image width in pixels",
        "Image height in pixels",
        "Number of cameras in dataset",
        "Number of points in 3D point cloud",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset metadata to unpack"
                }),
            }
        }

    def unpack(self, b2c_data):
        """
        Unpack dataset metadata into individual outputs.

        Args:
            b2c_data: B2C_COLMAP_METADATA from render or load nodes

        Returns:
            splat_scene: SPLAT_SCENE object (or None)
            splat_path: Path to PLY file (or empty string)
            width: Image width in pixels
            height: Image height in pixels
            camera_count: Number of cameras
            point_count: Number of points in point cloud
        """
        # Extract resolution (always present)
        width, height = b2c_data["resolution"]

        # Extract camera count (always present)
        camera_count = len(b2c_data["cameras"])

        # Extract point cloud count (always present)
        points_3d = b2c_data["points_3d"]
        positions, colors = points_3d
        point_count = len(positions)

        # Extract splat information (optional)
        splat_path = b2c_data.get("splat_path", "")
        if splat_path is None:
            splat_path = ""

        # Load splat scene if path exists
        splat_scene = None
        if splat_path and Path(splat_path).exists():
            try:
                from body2colmap.splat_scene import SplatScene
                splat_scene = SplatScene.from_ply(splat_path)
                logger.info(
                    f"[Body2COLMAP] Loaded splat: {len(splat_scene)} Gaussians, "
                    f"SH degree {splat_scene.sh_degree}"
                )
            except Exception as e:
                logger.warning(f"[Body2COLMAP] Failed to load splat from {splat_path}: {e}")
                splat_path = ""  # Clear path if load failed

        logger.info(
            f"[Body2COLMAP] Unpacked dataset: {width}x{height}, "
            f"{camera_count} cameras, {point_count} points, "
            f"splat: {'yes' if splat_scene else 'no'}"
        )

        return (splat_scene, splat_path, width, height, camera_count, point_count)
