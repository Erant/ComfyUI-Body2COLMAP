"""Save Gaussian Splat node - saves splat PLY files to output directory."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_next_numbered_file(base_path: Path, extension: str = ".ply") -> Path:
    """
    Find the next available numbered file following ComfyUI pattern.

    For base path 'output/splat.ply', finds all files matching 'output/splat_NNNNN.ply'
    where NNNNN is a decimal number, then returns 'output/splat_NNNNN+1.ply'.

    Args:
        base_path: Base file path (e.g., 'output/splat.ply')
        extension: File extension (default '.ply')

    Returns:
        Path to next numbered file (e.g., 'output/splat_00001.ply')
    """
    import re

    parent = base_path.parent
    stem = base_path.stem  # filename without extension

    # Pattern to match files like 'splat_00001.ply', 'splat_12345.ply', etc.
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(extension)}$")

    max_number = 0

    # Check if parent directory exists
    if parent.exists():
        # Find all matching numbered files
        for entry in parent.iterdir():
            if entry.is_file():
                match = pattern.match(entry.name)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)

    # Return next numbered file
    next_number = max_number + 1
    return parent / f"{stem}_{next_number:05d}{extension}"


class Body2COLMAP_SaveSplat:
    """Save Gaussian Splat PLY file to output directory."""

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
                "output_directory": ("STRING", {
                    "default": "splats",
                    "tooltip": "Output directory name (in output folder)"
                }),
                "filename": ("STRING", {
                    "default": "splat.ply",
                    "tooltip": "Output filename (e.g., splat.ply)"
                }),
                "auto_increment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-number files (splat_00001.ply, splat_00002.ply, etc.)"
                }),
            },
            "optional": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Optional metadata to get source splat path automatically"
                }),
                "source_filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Manual source PLY path (overrides b2c_data)"
                }),
            }
        }

    def save(self, splat_scene, output_directory, filename, auto_increment=True,
             b2c_data=None, source_filepath=""):
        """
        Save Gaussian Splat to output directory.

        Creates file at:
            output/directory/filename  (if auto_increment=False)
            output/directory/filename_NNNNN.ply  (if auto_increment=True)

        Args:
            splat_scene: SPLAT_SCENE object (for validation/info)
            output_directory: Output directory name (in output folder)
            filename: Output filename (e.g., splat.ply)
            auto_increment: If True, create numbered files
            b2c_data: Optional metadata containing splat_path
            source_filepath: Optional manual source path (overrides b2c_data)

        Returns:
            Absolute path to saved file
        """
        # Determine source PLY file path
        source_path = None

        if source_filepath and source_filepath.strip():
            # Manual source path provided
            source_path = Path(source_filepath.strip())
        elif b2c_data and "splat_path" in b2c_data:
            # Get from metadata
            splat_path = b2c_data["splat_path"]
            if splat_path:
                source_path = Path(splat_path)

        if source_path is None or not source_path.exists():
            if source_path is None:
                raise ValueError(
                    "No source splat path found. "
                    "Provide either 'b2c_data' with splat_path or 'source_filepath'."
                )
            else:
                raise FileNotFoundError(f"Source splat file not found: {source_path}")

        # Build output path
        output_dir = Path("output") / output_directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure filename has .ply extension
        if not filename.endswith('.ply'):
            filename = filename + '.ply'

        base_path = output_dir / filename

        if auto_increment:
            # Find next available numbered file
            output_path = get_next_numbered_file(base_path, extension='.ply')
        else:
            # Use exact filename
            output_path = base_path

        # Copy PLY file
        logger.info(f"[Body2COLMAP] Copying splat from {source_path} to {output_path}")
        shutil.copy(source_path, output_path)

        logger.info(
            f"[Body2COLMAP] Saved Gaussian Splat: {len(splat_scene)} Gaussians, "
            f"SH degree {splat_scene.sh_degree}"
        )

        print(f"[Body2COLMAP] Gaussian Splat saved to: {output_path}")
        print(f"[Body2COLMAP] - {len(splat_scene)} Gaussians")
        print(f"[Body2COLMAP] - SH degree: {splat_scene.sh_degree}")

        return (str(output_path.absolute()),)
