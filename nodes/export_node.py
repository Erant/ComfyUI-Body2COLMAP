"""Export node for Body2COLMAP - exports COLMAP sparse reconstruction format."""

from pathlib import Path
import cv2
import folder_paths
import numpy as np
import re
import torch
from body2colmap.exporter import ColmapExporter
from ..core.comfy_utils import comfy_to_cv2


def get_next_numbered_directory(base_path: Path) -> Path:
    """
    Find the next available numbered directory following ComfyUI Save Image pattern.

    For base path 'output/splat', finds all directories matching 'output/splat_NNNNN'
    where NNNNN is a decimal number, then returns 'output/splat_NNNNN+1'.

    Args:
        base_path: Base directory path (e.g., 'output/splat')

    Returns:
        Path to next numbered directory (e.g., 'output/splat_00001')
    """
    parent = base_path.parent
    base_name = base_path.name

    # Pattern to match directories like 'splat_00001', 'splat_12345', etc.
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")

    max_number = 0

    # Check if parent directory exists
    if parent.exists():
        # Find all matching numbered directories
        for entry in parent.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)

    # Return next numbered directory
    next_number = max_number + 1
    return parent / f"{base_name}_{next_number:05d}"


class Body2COLMAP_ExportCOLMAP:
    """Export COLMAP sparse reconstruction format for 3DGS training."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True  # Terminal node - produces file output
    OUTPUT_TOOLTIPS = ("Path to the output directory containing COLMAP files",)

    # Tell ComfyUI to collect all batch outputs into lists
    INPUT_IS_LIST = {
        "images": True,
        "masks": True,
        "normal_maps": True,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA",),
                "output_directory": ("STRING", {
                    "default": "colmap",
                    "tooltip": "Base directory name (in output folder)"
                }),
                "auto_increment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-number directories (colmap_00001, colmap_00002, etc.)"
                }),
                "merge_batches": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Merge batched inputs into single dataset (enable when loading with batch_size > 0)"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "normal_maps": ("IMAGE", {
                    "tooltip": (
                        "Optional per-frame normal maps, one per image. Written to a "
                        "'normals/' directory beside the images, matched by filename stem. "
                        "Ignored if images is not also provided."
                    )
                }),
            }
        }

    def export(self, b2c_data, output_directory, auto_increment=True, merge_batches=False, images=None, masks=None, normal_maps=None):
        """
        Export COLMAP format files.

        Creates (flat structure matching body2colmap):
            output/directory/  (if auto_increment=False)
            output/directory_NNNNN/  (if auto_increment=True)
            ├── frame_00001_.png (RGBA if masks provided, RGB otherwise)
            ├── frame_00002_.png
            ├── ...
            ├── cameras.txt   (camera intrinsics)
            ├── images.txt    (camera extrinsics per image)
            └── points3D.txt  (initial point cloud)

        Args:
            b2c_data: B2C_COLMAP_METADATA from render nodes or LoadDataset
            output_directory: Base directory name (in output folder)
            auto_increment: If True, create numbered directories (colmap_00001, etc.)
            merge_batches: If True, merge batched inputs into single dataset
            images: Optional ComfyUI IMAGE tensor or List[IMAGE] when batched
            masks: Optional ComfyUI MASK tensor or List[MASK] when batched
            normal_maps: Optional ComfyUI IMAGE tensor or List[IMAGE] when batched, one per
                image. Written to a 'normals/' directory beside the images.

        Note:
            Point cloud must be pre-sampled in render nodes and included in b2c_data.
        """
        # Unwrap scalar parameters if they come as lists (happens when INPUT_IS_LIST is set)
        # When INPUT_IS_LIST is present, ComfyUI passes all inputs as lists in batched contexts
        if isinstance(b2c_data, list):
            b2c_data = b2c_data[0]
        if isinstance(output_directory, list):
            output_directory = output_directory[0]
        if isinstance(auto_increment, list):
            auto_increment = auto_increment[0]
        if isinstance(merge_batches, list):
            merge_batches = merge_batches[0]

        # Handle batch merging
        if merge_batches:
            # Concatenate all batches into single tensor
            if images is not None:
                if isinstance(images, list) and len(images) > 1:
                    images = torch.cat(images, dim=0)
                elif isinstance(images, list):
                    images = images[0]  # Single batch

            if masks is not None:
                if isinstance(masks, list) and len(masks) > 1:
                    masks = torch.cat(masks, dim=0)
                elif isinstance(masks, list):
                    masks = masks[0]  # Single batch

            if normal_maps is not None:
                if isinstance(normal_maps, list) and len(normal_maps) > 1:
                    normal_maps = torch.cat(normal_maps, dim=0)
                elif isinstance(normal_maps, list):
                    normal_maps = normal_maps[0]  # Single batch
        else:
            # Extract single batch (backward compatible)
            if images is not None and isinstance(images, list):
                if len(images) > 1:
                    raise ValueError(
                        f"Received {len(images)} batches but merge_batches=False. "
                        "Enable merge_batches or disable batching in Load Dataset (set batch_size=0)"
                    )
                images = images[0]

            if masks is not None and isinstance(masks, list):
                if len(masks) > 1:
                    raise ValueError(
                        f"Received {len(masks)} mask batches but merge_batches=False. "
                        "Enable merge_batches or disable batching in Load Dataset (set batch_size=0)"
                    )
                masks = masks[0]

            if normal_maps is not None and isinstance(normal_maps, list):
                if len(normal_maps) > 1:
                    raise ValueError(
                        f"Received {len(normal_maps)} normal map batches but merge_batches=False. "
                        "Enable merge_batches or disable batching in Load Dataset (set batch_size=0)"
                    )
                normal_maps = normal_maps[0]

        # Every training view needs its own normal map, or downstream tools would
        # silently pair them up by index against the wrong frames.
        if normal_maps is not None and images is not None and len(normal_maps) != len(images):
            raise ValueError(
                f"Normal map count ({len(normal_maps)}) does not match image count "
                f"({len(images)}). Every image needs a matching normal map."
            )

        # Extract data from b2c_data
        cameras = b2c_data["cameras"]
        image_names = b2c_data["image_names"]
        points_3d = b2c_data["points_3d"]
        width, height = b2c_data["resolution"]

        # Build output path (absolute paths used as-is, relative paths resolve under ComfyUI output)
        dir_path = Path(output_directory)
        base_path = dir_path if dir_path.is_absolute() else Path(folder_paths.get_output_directory()) / output_directory

        if auto_increment:
            # Create numbered directory
            output_path = get_next_numbered_directory(base_path)
        else:
            # Use exact directory
            output_path = base_path

        output_path.mkdir(parents=True, exist_ok=True)

        # Save images if provided
        alpha_channel = None
        if images is not None:
            # Convert ComfyUI images to CV2 format
            cv2_images = comfy_to_cv2(images)

            # If masks provided, combine with images as alpha channel
            if masks is not None:
                # Convert masks from ComfyUI format [B, H, W] float [0,1] to [B, H, W] uint8 [0,255]
                # Note: ComfyUI MASK is inverted (1.0 = background), so we invert back for alpha channel
                masks_np = masks.cpu().numpy()
                alpha_channel = ((1.0 - masks_np) * 255).astype(np.uint8)  # Invert back for alpha

                # Combine RGB with alpha
                for i, (img, filename) in enumerate(zip(cv2_images, image_names)):
                    alpha = alpha_channel[i]  # [H, W]

                    if img.shape[-1] == 4:
                        # Already BGRA - replace existing alpha channel
                        rgba = img.copy()
                        rgba[..., 3] = alpha
                    elif img.shape[-1] == 3:
                        # BGR - add alpha channel
                        rgba = np.dstack([img, alpha])  # [H, W, 4] - BGRA
                    else:
                        raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")

                    img_path = output_path / filename
                    cv2.imwrite(str(img_path), rgba)
            else:
                # Save RGB only
                for img, filename in zip(cv2_images, image_names):
                    img_path = output_path / filename
                    cv2.imwrite(str(img_path), img)

        # Save normal maps if provided, to a 'normals/' directory beside the images
        # (matches the layout brush's normal-map supervision expects).
        if normal_maps is not None and images is not None:
            normals_dir = output_path / "normals"
            normals_dir.mkdir(exist_ok=True)

            cv2_normals = comfy_to_cv2(normal_maps)

            for i, (normal, filename) in enumerate(zip(cv2_normals, image_names)):
                # A normal map that carries its own alpha keeps it; otherwise borrow the
                # RGB frame's mask, which is the foreground the loss is restricted to.
                if normal.shape[-1] == 3 and alpha_channel is not None:
                    out = np.dstack([normal, alpha_channel[i]])  # [H, W, 4] - BGRA
                else:
                    out = normal

                # Match Brush's expectation: same stem as the RGB frame, forced to PNG.
                normal_path = normals_dir / Path(filename).with_suffix(".png").name
                cv2.imwrite(str(normal_path), out)

            print(f"[Body2COLMAP] - {len(cv2_normals)} normal map files")

        # Create COLMAP exporter with pre-sampled point cloud
        exporter = ColmapExporter(
            cameras=cameras,
            image_names=image_names,
            points_3d=points_3d
        )

        # Export COLMAP files to same directory as images
        exporter.export(output_dir=output_path)

        print(f"[Body2COLMAP] Exported COLMAP files to: {output_path}")
        print(f"[Body2COLMAP] - cameras.txt: {len(cameras)} cameras")
        print(f"[Body2COLMAP] - images.txt: {len(cameras)} images")
        print(f"[Body2COLMAP] - points3D.txt: {len(points_3d[0])} points")

        if images is not None:
            format_str = "RGBA" if masks is not None else "RGB"
            print(f"[Body2COLMAP] - {len(cv2_images)} {format_str} image files")

        return (str(output_path.absolute()),)
