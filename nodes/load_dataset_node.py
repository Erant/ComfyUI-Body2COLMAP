"""Load dataset node - loads Body2COLMAP datasets from disk."""

import json
import logging
from pathlib import Path
import numpy as np
import cv2
import torch

from body2colmap.camera import Camera

logger = logging.getLogger(__name__)


def deserialize_camera(camera_data: dict, resolution: tuple) -> Camera:
    """
    Reconstruct body2colmap Camera object from JSON dict.

    Args:
        camera_data: Dict with intrinsics and extrinsics
        resolution: (width, height) tuple

    Returns:
        body2colmap Camera object
    """
    intrinsics = camera_data["intrinsics"]
    extrinsics = camera_data["extrinsics"]

    # Reconstruct Camera with all parameters
    camera = Camera(
        focal_length=(intrinsics["fx"], intrinsics["fy"]),
        image_size=resolution,
        principal_point=(intrinsics["cx"], intrinsics["cy"]),
        position=np.array(extrinsics["position"], dtype=np.float32),
        rotation=np.array(extrinsics["rotation"], dtype=np.float32)
    )

    return camera


def cv2_to_comfy_image(images_bgr: list) -> torch.Tensor:
    """
    Convert list of CV2 BGR images to ComfyUI IMAGE format.

    Args:
        images_bgr: List of CV2 images [H, W, 3] BGR uint8

    Returns:
        ComfyUI IMAGE tensor [B, H, W, 3] RGB float32 [0,1]
    """
    # Convert BGR to RGB and normalize
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]
    images_float = [img.astype(np.float32) / 255.0 for img in images_rgb]

    # Stack into batch tensor
    images_tensor = torch.from_numpy(np.stack(images_float, axis=0))

    return images_tensor


def extract_alpha_to_comfy_mask(images_bgra: list) -> torch.Tensor:
    """
    Extract alpha channel from BGRA images to ComfyUI MASK format.

    Args:
        images_bgra: List of CV2 images [H, W, 4] BGRA uint8

    Returns:
        ComfyUI MASK tensor [B, H, W] float32 [0,1]
        Note: ComfyUI MASK is inverted (1.0 = background)
    """
    # Extract alpha channel (index 3)
    alphas = [img[:, :, 3].astype(np.float32) / 255.0 for img in images_bgra]

    # Invert for ComfyUI convention (1.0 = background)
    masks = [1.0 - alpha for alpha in alphas]

    # Stack into batch tensor
    masks_tensor = torch.from_numpy(np.stack(masks, axis=0))

    return masks_tensor


class Body2COLMAP_LoadDataset:
    """
    Load Body2COLMAP dataset from disk.

    Supports auto-incrementing/decrementing index for batch processing multiple datasets.
    State is tracked per-node instance to support batch queue mode.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "load"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("b2c_data", "images", "masks", "reference_image")
    OUTPUT_TOOLTIPS = (
        "Body2COLMAP dataset metadata (connect to ExportCOLMAP or SaveDataset)",
        "Batch of loaded images",
        "Batch of alpha masks",
        "Reference image for preview (empty if not saved)"
    )

    # Class-level state for tracking index across batch executions
    # Key: node_id -> current_index
    _batch_state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "b2c_dataset",
                    "tooltip": "Base directory name (in output folder)"
                }),
                "index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99999,
                    "tooltip": "Starting dataset index (-1 = use directory as-is, >=0 = append _NNNNN)"
                }),
                "index_control": (["fixed", "increment", "decrement"], {
                    "default": "fixed",
                    "tooltip": "Auto-increment behavior: fixed=use index as-is, increment=+1 per run, decrement=-1 per run. State tracked internally for batch mode. Switch to 'fixed' to reset."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    @classmethod
    def IS_CHANGED(cls, directory, index=-1, index_control="fixed"):
        """
        Force re-execution when using increment/decrement mode.

        ComfyUI caches node results when inputs haven't changed. Since JavaScript
        updates the index widget after execution, we need to force re-execution
        when in increment/decrement mode to prevent cached results.
        """
        import time
        if index_control != "fixed":
            # Return unique value to force re-execution
            return float(time.time())
        # Return stable value when fixed to allow caching
        return f"{directory}_{index}"

    def load(self, directory, index=-1, index_control="fixed", unique_id=None):
        """
        Load Body2COLMAP dataset from disk.

        Reads from directory structure:
            output/directory/  (if index=-1)
            output/directory_NNNNN/  (if index>=0)
            ├── frame_00001_.png
            ├── frame_00002_.png
            ├── ...
            ├── reference.png (optional)
            ├── metadata.json
            └── pointcloud.npz

        Args:
            directory: Base directory name (in output folder)
            index: Starting dataset index (-1 = exact path, >=0 = append _NNNNN)
            index_control: Auto-increment control (state tracked internally)
                          - fixed: Always use index widget value, clear any state
                          - increment: Start at index, then +1 per execution
                          - decrement: Start at index, then -1 per execution
                          Note: Widget shows starting value only. Actual index tracked internally.
                          To reset: switch to "fixed", change index, switch back.
            unique_id: Node ID (hidden parameter, auto-provided by ComfyUI)

        Returns:
            b2c_data: B2C_COLMAP_METADATA
            images: ComfyUI IMAGE tensor
            masks: ComfyUI MASK tensor
            reference_image: ComfyUI IMAGE tensor (or empty if not present)
        """
        # Determine actual index to use (handles batch mode)
        if index_control == "fixed":
            # Use index as-is, no state tracking
            # Clear any cached state to allow fresh start if switching back to increment/decrement
            if unique_id in self._batch_state:
                del self._batch_state[unique_id]
            actual_index = index
        else:
            # Use batch state to track index across executions (keyed by node ID)
            # Note: Once initialized, state persists and widget changes are ignored.
            # To reset: switch to "fixed" mode, change index, then switch back.
            if unique_id not in self._batch_state:
                # First execution for this node - use the widget value
                self._batch_state[unique_id] = index
            actual_index = self._batch_state[unique_id]

        # Build full path
        if actual_index == -1:
            # Use directory as-is
            dataset_path = Path("output") / directory
        else:
            # Append _NNNNN to directory
            numbered_dir = f"{directory}_{actual_index:05d}"
            dataset_path = Path("output") / numbered_dir

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        logger.info(f"[Body2COLMAP] Loading dataset from: {dataset_path}")

        # Load metadata.json
        metadata_path = dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {dataset_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        version = metadata.get("version", "1.0")
        resolution = tuple(metadata["resolution"])
        camera_data_list = metadata["cameras"]

        logger.info(f"[Body2COLMAP] Dataset version: {version}")
        logger.info(f"[Body2COLMAP] Resolution: {resolution[0]}x{resolution[1]}")

        # Reconstruct Camera objects
        cameras = []
        image_names = []

        for cam_data in camera_data_list:
            image_names.append(cam_data["image_name"])
            camera = deserialize_camera(cam_data, resolution)
            cameras.append(camera)

        logger.info(f"[Body2COLMAP] Reconstructed {len(cameras)} cameras")

        # Load point cloud
        pointcloud_path = dataset_path / "pointcloud.npz"
        if not pointcloud_path.exists():
            raise FileNotFoundError(f"pointcloud.npz not found in {dataset_path}")

        pointcloud_data = np.load(pointcloud_path)
        positions = pointcloud_data["positions"]
        colors = pointcloud_data["colors"]

        logger.info(f"[Body2COLMAP] Loaded point cloud ({len(positions)} points)")

        # Load images
        loaded_images = []
        has_alpha = False

        for image_name in image_names:
            img_path = dataset_path / image_name
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Load with alpha channel support
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            loaded_images.append(img)

            # Check if first image has alpha
            if len(loaded_images) == 1:
                has_alpha = (img.shape[2] == 4) if len(img.shape) == 3 else False

        logger.info(f"[Body2COLMAP] Loaded {len(loaded_images)} images")

        # Extract RGB and masks
        if has_alpha:
            # Images have alpha channel - extract it for masks
            images_bgr = [img[:, :, :3] for img in loaded_images]
            images_tensor = cv2_to_comfy_image(images_bgr)
            masks_tensor = extract_alpha_to_comfy_mask(loaded_images)
            logger.info("[Body2COLMAP] Extracted alpha channel as masks")
        else:
            # No alpha channel - create empty masks
            images_tensor = cv2_to_comfy_image(loaded_images)
            # Create masks of all zeros (foreground in ComfyUI convention)
            h, w = loaded_images[0].shape[:2]
            masks_tensor = torch.zeros((len(loaded_images), h, w), dtype=torch.float32)
            logger.info("[Body2COLMAP] No alpha channel, created empty masks")

        # Load reference image if exists
        reference_path = dataset_path / "reference.png"
        if reference_path.exists():
            ref_img = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
            # Extract RGB if has alpha
            if len(ref_img.shape) == 3 and ref_img.shape[2] == 4:
                ref_img = ref_img[:, :, :3]
            reference_tensor = cv2_to_comfy_image([ref_img])
            logger.info("[Body2COLMAP] Loaded reference image")
        else:
            # Create empty 1x1 image as placeholder
            reference_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            logger.info("[Body2COLMAP] No reference image found")

        # Package metadata
        b2c_data = {
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": (positions, colors),
            "resolution": resolution,
        }

        print(f"[Body2COLMAP] Loaded dataset from: {dataset_path}")
        print(f"[Body2COLMAP] - {len(image_names)} images")
        print(f"[Body2COLMAP] - {len(cameras)} cameras")
        print(f"[Body2COLMAP] - {len(positions)} points")
        if reference_path.exists():
            print("[Body2COLMAP] - reference.png")

        # Update batch state for next execution
        next_index = actual_index
        if index_control == "increment":
            next_index = actual_index + 1
            self._batch_state[unique_id] = next_index
        elif index_control == "decrement":
            next_index = actual_index - 1
            self._batch_state[unique_id] = next_index

        # Return UI update with next index for JavaScript to display
        return {
            "ui": {"index": [next_index]},
            "result": (b2c_data, images_tensor, masks_tensor, reference_tensor)
        }
