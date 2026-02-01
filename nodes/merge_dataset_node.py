"""Merge dataset node - combines multiple Body2COLMAP datasets."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Body2COLMAP_MergeDatasets:
    """Merge multiple Body2COLMAP datasets into a single dataset."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "merge"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Merged B2C_COLMAP_METADATA",
        "Concatenated images from all datasets",
        "Concatenated masks from all datasets"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # First dataset (always present)
                "b2c_data_1": ("B2C_COLMAP_METADATA",),
                "images_1": ("IMAGE",),
                "masks_1": ("MASK",),

                # Point cloud handling
                "pointcloud_mode": (["first", "merge", "resample"], {
                    "default": "first",
                    "tooltip": "How to combine point clouds: first=use first dataset, merge=concatenate all, resample=combine and randomly sample N points"
                }),
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample (only used when pointcloud_mode=resample)"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image for the merged dataset"
                }),
            }
        }

    def merge(self, b2c_data_1, images_1, masks_1, pointcloud_mode="first",
              pointcloud_samples=10000, reference_image=None, **kwargs):
        """
        Merge multiple datasets into a single dataset.

        Collects all b2c_data_N, images_N, masks_N from kwargs and combines them.
        The JavaScript extension adds these dynamically as users connect inputs.

        Args:
            b2c_data_1: First dataset metadata
            images_1: First dataset images
            masks_1: First dataset masks
            pointcloud_mode: How to combine point clouds
            pointcloud_samples: Number of points for resampling
            reference_image: Optional reference image (not used in merge, just for SaveDataset)
            **kwargs: Additional datasets (b2c_data_2, images_2, masks_2, etc.)

        Returns:
            merged_b2c_data: Combined metadata
            merged_images: Concatenated images
            merged_masks: Concatenated masks
        """
        # Collect all datasets
        datasets = [(b2c_data_1, images_1, masks_1)]

        i = 2
        while f"b2c_data_{i}" in kwargs:
            b2c_data = kwargs[f"b2c_data_{i}"]
            images = kwargs[f"images_{i}"]
            masks = kwargs[f"masks_{i}"]
            datasets.append((b2c_data, images, masks))
            i += 1

        logger.info(f"[Body2COLMAP] Merging {len(datasets)} datasets")

        # Validate resolution consistency
        first_resolution = datasets[0][0]["resolution"]
        for idx, (b2c_data, _, _) in enumerate(datasets):
            if b2c_data["resolution"] != first_resolution:
                raise ValueError(
                    f"Dataset {idx+1} has resolution {b2c_data['resolution']}, "
                    f"but first dataset has {first_resolution}. "
                    "All datasets must have the same resolution."
                )

        # Merge cameras and image names
        all_cameras = []
        all_image_names = []
        frame_counter = 1

        for b2c_data, _, _ in datasets:
            all_cameras.extend(b2c_data["cameras"])
            # Renumber frames sequentially
            for _ in b2c_data["image_names"]:
                all_image_names.append(f"frame_{frame_counter:05d}_.png")
                frame_counter += 1

        logger.info(f"[Body2COLMAP] Merged {len(all_cameras)} cameras")

        # Merge images and masks
        all_images = [images for _, images, _ in datasets]
        all_masks = [masks for _, _, masks in datasets]

        merged_images = torch.cat(all_images, dim=0)
        merged_masks = torch.cat(all_masks, dim=0)

        logger.info(f"[Body2COLMAP] Merged {len(merged_images)} images")

        # Handle point cloud based on mode
        if pointcloud_mode == "first":
            # Use first dataset's point cloud
            merged_points_3d = datasets[0][0]["points_3d"]
            logger.info("[Body2COLMAP] Using first dataset's point cloud")

        elif pointcloud_mode == "merge":
            # Concatenate all point clouds
            all_positions = []
            all_colors = []

            for b2c_data, _, _ in datasets:
                positions, colors = b2c_data["points_3d"]
                all_positions.append(positions)
                all_colors.append(colors)

            merged_positions = np.concatenate(all_positions, axis=0)
            merged_colors = np.concatenate(all_colors, axis=0)
            merged_points_3d = (merged_positions, merged_colors)

            logger.info(f"[Body2COLMAP] Merged {len(merged_positions)} points from all datasets")

        elif pointcloud_mode == "resample":
            # Combine all point clouds, then randomly sample N points
            all_positions = []
            all_colors = []

            for b2c_data, _, _ in datasets:
                positions, colors = b2c_data["points_3d"]
                all_positions.append(positions)
                all_colors.append(colors)

            combined_positions = np.concatenate(all_positions, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)

            # Randomly sample pointcloud_samples points
            total_points = len(combined_positions)
            if total_points <= pointcloud_samples:
                # Use all points if we have fewer than requested
                merged_positions = combined_positions
                merged_colors = combined_colors
                logger.info(f"[Body2COLMAP] Using all {total_points} points (less than requested {pointcloud_samples})")
            else:
                # Random sampling without replacement
                indices = np.random.choice(total_points, size=pointcloud_samples, replace=False)
                merged_positions = combined_positions[indices]
                merged_colors = combined_colors[indices]
                logger.info(f"[Body2COLMAP] Resampled {pointcloud_samples} points from {total_points} total points")

            merged_points_3d = (merged_positions, merged_colors)
        else:
            raise ValueError(f"Unknown pointcloud_mode: {pointcloud_mode}")

        # Create merged metadata
        merged_b2c_data = {
            "cameras": all_cameras,
            "image_names": all_image_names,
            "points_3d": merged_points_3d,
            "resolution": first_resolution,
        }

        print(f"[Body2COLMAP] Merged {len(datasets)} datasets:")
        print(f"[Body2COLMAP] - {len(all_cameras)} total cameras")
        print(f"[Body2COLMAP] - {len(merged_images)} total images")
        print(f"[Body2COLMAP] - {len(merged_points_3d[0])} points in merged point cloud")

        return (merged_b2c_data, merged_images, merged_masks)
