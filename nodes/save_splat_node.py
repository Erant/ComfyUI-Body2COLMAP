"""Save Gaussian Splat node - writes a SplatScene to a PLY file."""

import logging
from pathlib import Path

import folder_paths

logger = logging.getLogger(__name__)


class Body2COLMAP_SaveSplat:
    """Save a Gaussian Splat scene to a PLY file.

    Writes the splat to disk using body2colmap's SplatScene.to_ply().
    Relative paths are resolved under ComfyUI's output directory;
    absolute paths are used as-is.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("Absolute path to the saved PLY file",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "splat_scene": ("SPLAT_SCENE", {
                    "tooltip": "Gaussian Splat scene to save"
                }),
                "filename": ("STRING", {
                    "default": "splat.ply",
                    "multiline": False,
                    "tooltip": (
                        "Output filename or path.  Relative paths are "
                        "placed inside ComfyUI's output folder; absolute "
                        "paths are used as-is."
                    )
                }),
            },
        }

    def save(self, splat_scene, filename):
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = Path(folder_paths.get_output_directory()) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Body2COLMAP] Saving Gaussian Splat ({len(splat_scene)} Gaussians) to: {filepath}")
        splat_scene.to_ply(str(filepath))
        logger.info(f"[Body2COLMAP] Splat saved: {filepath}")

        return (str(filepath.absolute()),)
