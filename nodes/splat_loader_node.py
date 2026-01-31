"""Splat loader node - loads Gaussian Splats from PLY files."""

import logging

logger = logging.getLogger(__name__)


class Body2COLMAP_LoadSplat:
    """Load Gaussian Splat from PLY file (trained 3DGS output)."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "load"
    RETURN_TYPES = ("SPLAT_SCENE",)
    RETURN_NAMES = ("splat_scene",)
    OUTPUT_TOOLTIPS = (
        "Gaussian Splat scene loaded from PLY (connect to RenderSplat)",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to Gaussian Splat PLY file (output from 3DGS training)"
                })
            }
        }

    def load(self, filepath):
        """
        Load Gaussian Splat from PLY file.

        Args:
            filepath: Path to .ply file containing trained Gaussian Splat

        Returns:
            Tuple containing SplatScene object
        """
        # Import body2colmap SplatScene
        try:
            from body2colmap.splat_scene import SplatScene
        except ImportError as e:
            raise ImportError(
                "Failed to import body2colmap.splat_scene. "
                "Make sure body2colmap is installed with splat support: "
                "pip install body2colmap[splat]"
            ) from e

        # Load PLY file
        logger.info(f"[Body2COLMAP] Loading Gaussian Splat from: {filepath}")
        scene = SplatScene.from_ply(filepath)

        logger.info(
            f"[Body2COLMAP] Splat loaded: {len(scene)} Gaussians, "
            f"SH degree {scene.sh_degree}"
        )

        return (scene,)
