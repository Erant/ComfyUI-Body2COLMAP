"""Save dataset node - saves Body2COLMAP datasets to disk."""

import json
import logging
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch

from ..core.comfy_utils import comfy_to_cv2

logger = logging.getLogger(__name__)


def get_next_numbered_directory(base_path: Path) -> Path:
    """
    Find the next available numbered directory following ComfyUI Save Image pattern.

    For base path 'output/dataset', finds all directories matching 'output/dataset_NNNNN'
    where NNNNN is a decimal number, then returns 'output/dataset_NNNNN+1'.

    Args:
        base_path: Base directory path (e.g., 'output/dataset')

    Returns:
        Path to next numbered directory (e.g., 'output/dataset_00001')
    """
    import re

    parent = base_path.parent
    base_name = base_path.name

    # Pattern to match directories like 'dataset_00001', 'dataset_12345', etc.
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


def serialize_camera(camera) -> dict:
    """
    Serialize body2colmap Camera object to JSON-compatible dict.

    Args:
        camera: body2colmap Camera object

    Returns:
        Dict with intrinsics and extrinsics
    """
    return {
        "intrinsics": {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
        },
        "extrinsics": {
            "rotation": camera.rotation.tolist(),  # 3x3 rotation matrix (camera-to-world)
            "position": camera.position.tolist(),  # 3D position vector
        }
    }


class Body2COLMAP_SaveDataset:
    """Save Body2COLMAP dataset to disk in intermediary format."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory_path",)
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("Path to the saved dataset directory",)

    # Tell ComfyUI to collect all batch outputs into lists
    INPUT_IS_LIST = {
        "images": True,
        "masks": True,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA",),
                "images": ("IMAGE",),
                "output_directory": ("STRING", {
                    "default": "b2c_dataset",
                    "tooltip": "Base directory name (in output folder)"
                }),
                "auto_increment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-number directories (dataset_00001, dataset_00002, etc.)"
                }),
                "merge_batches": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Merge batched inputs into single dataset (enable when loading with batch_size > 0)"
                }),
            },
            "optional": {
                "masks": ("MASK",),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image saved as reference.png for preview"
                }),
            }
        }

    def save(self, b2c_data, images, output_directory, auto_increment=True, merge_batches=False, masks=None, reference_image=None):
        """
        Save Body2COLMAP dataset to disk.

        Creates directory structure:
            output/directory/  (if auto_increment=False)
            output/directory_NNNNN/  (if auto_increment=True)
            ├── frame_00001_.png
            ├── frame_00002_.png
            ├── ...
            ├── reference.png (optional)
            ├── splat.ply (optional, if splat_path in b2c_data)
            ├── metadata.json
            └── pointcloud.npz

        Args:
            b2c_data: B2C_COLMAP_METADATA from render nodes
            images: ComfyUI IMAGE tensor or List[IMAGE] when batched
            output_directory: Base directory name (in output folder)
            auto_increment: If True, create numbered directories (dataset_00001, etc.)
            merge_batches: If True, merge batched inputs into single dataset
            masks: Optional ComfyUI MASK tensor or List[MASK] when batched
            reference_image: Optional reference image for preview

        Returns:
            Absolute path to created directory
        """
        # Handle batch merging
        if merge_batches:
            # Concatenate all batches into single tensor
            if isinstance(images, list) and len(images) > 1:
                images = torch.cat(images, dim=0)
                logger.info(f"[Body2COLMAP] Merged {len(images)} image batches")
            elif isinstance(images, list):
                images = images[0]  # Single batch

            if masks is not None:
                if isinstance(masks, list) and len(masks) > 1:
                    masks = torch.cat(masks, dim=0)
                    logger.info(f"[Body2COLMAP] Merged {len(masks)} mask batches")
                elif isinstance(masks, list):
                    masks = masks[0]  # Single batch
        else:
            # Extract single batch (backward compatible)
            if isinstance(images, list):
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

        # Build output path
        base_path = Path("output") / output_directory

        if auto_increment:
            # Create numbered directory
            output_path = get_next_numbered_directory(base_path)
        else:
            # Use exact directory
            output_path = base_path

        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Body2COLMAP] Saving dataset to: {output_path}")

        # Extract metadata
        cameras = b2c_data["cameras"]
        image_names = b2c_data["image_names"]
        points_3d = b2c_data["points_3d"]
        resolution = b2c_data["resolution"]

        # Convert images to CV2 format
        cv2_images = comfy_to_cv2(images)

        # Save images (with alpha channel if masks provided)
        if masks is not None:
            # Convert masks from ComfyUI format [B, H, W] float [0,1] to [B, H, W] uint8 [0,255]
            # Note: ComfyUI MASK is inverted (1.0 = background), so we invert back for alpha channel
            masks_np = masks.cpu().numpy()
            alpha_channel = ((1.0 - masks_np) * 255).astype(np.uint8)

            # Save RGBA
            for i, (img, filename) in enumerate(zip(cv2_images, image_names)):
                alpha = alpha_channel[i]  # [H, W]

                # Check if image already has alpha channel (4 channels)
                if img.shape[-1] == 4:
                    # Replace existing alpha channel with our mask
                    rgba = img.copy()
                    rgba[..., 3] = alpha
                elif img.shape[-1] == 3:
                    # Add alpha channel to BGR image
                    rgba = np.dstack([img, alpha])  # [H, W, 4] - BGRA
                else:
                    raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")

                img_path = output_path / filename
                cv2.imwrite(str(img_path), rgba)
        else:
            # Save images without modifying alpha channel (RGB or RGBA as-is)
            for img, filename in zip(cv2_images, image_names):
                img_path = output_path / filename
                # Image can be BGR (3 channels) or BGRA (4 channels)
                cv2.imwrite(str(img_path), img)

        logger.info(f"[Body2COLMAP] Saved {len(cv2_images)} images")

        # Save reference image if provided
        if reference_image is not None:
            ref_cv2 = comfy_to_cv2(reference_image)
            # reference_image is IMAGE tensor, might be batch of 1
            ref_img = ref_cv2[0] if len(ref_cv2) > 0 else ref_cv2
            ref_path = output_path / "reference.png"
            cv2.imwrite(str(ref_path), ref_img)
            logger.info("[Body2COLMAP] Saved reference image")

        # Serialize cameras to metadata.json
        metadata = {
            "version": "1.0",
            "resolution": list(resolution),
            "cameras": [
                {
                    "image_name": name,
                    **serialize_camera(cam)
                }
                for name, cam in zip(image_names, cameras)
            ]
        }

        # Save splat if available in b2c_data
        source_splat = b2c_data.get("splat_path")
        if source_splat and Path(source_splat).exists():
            splat_filename = "splat.ply"
            splat_path = output_path / splat_filename
            shutil.copy(source_splat, splat_path)
            logger.info(f"[Body2COLMAP] Saved splat to {splat_path}")

            # Update metadata with splat filename
            metadata["splat_filename"] = splat_filename

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Body2COLMAP] Saved metadata ({len(cameras)} cameras)")

        # Save point cloud to pointcloud.npz
        positions, colors = points_3d
        pointcloud_path = output_path / "pointcloud.npz"
        np.savez_compressed(
            pointcloud_path,
            positions=positions,
            colors=colors
        )

        logger.info(f"[Body2COLMAP] Saved point cloud ({len(positions)} points)")

        print(f"[Body2COLMAP] Dataset saved to: {output_path}")
        print(f"[Body2COLMAP] - {len(image_names)} images")
        print(f"[Body2COLMAP] - {len(cameras)} cameras")
        print(f"[Body2COLMAP] - {len(positions)} points")
        if reference_image is not None:
            print("[Body2COLMAP] - reference.png")
        if metadata.get("splat_filename"):
            print("[Body2COLMAP] - splat.ply")

        return (str(output_path.absolute()),)
