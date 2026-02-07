"""Drop views node - remove specific views from a dataset for training/testing splits."""

import logging
import re

import torch

logger = logging.getLogger(__name__)


def parse_view_indices(spec: str) -> set:
    """Parse a view specification string into a set of 1-based indices.

    Supports comma-separated values and ranges:
        "1,2,3"     -> {1, 2, 3}
        "9-40"      -> {9, 10, ..., 40}
        "1,2,3,9-40" -> {1, 2, 3, 9, 10, ..., 40}
    """
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start ({start}) is greater than end ({end})"
                )
            indices.update(range(start, end + 1))
        elif re.fullmatch(r"\d+", part):
            indices.add(int(part))
        else:
            raise ValueError(
                f"Invalid view specification '{part}'. "
                "Use comma-separated integers and ranges, e.g. '1,2,3,9-40'"
            )
    return indices


class Body2COLMAP_DropViews:
    """Remove specific views from a dataset by index."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "drop"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Dataset metadata with specified views removed",
        "Image batch with specified views removed",
        "Mask batch with specified views removed",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset metadata to filter"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image batch to filter"
                }),
                "masks": ("MASK", {
                    "tooltip": "Mask batch to filter"
                }),
                "views_to_drop": ("STRING", {
                    "default": "",
                    "tooltip": "1-based view indices to remove, e.g. '1,2,3,9-40'"
                }),
            },
        }

    def drop(self, b2c_data, images, masks, views_to_drop):
        drop_indices = parse_view_indices(views_to_drop)

        if not drop_indices:
            logger.info("[Body2COLMAP] DropViews: no views specified, passing through unchanged")
            return (b2c_data, images, masks)

        n_views = len(b2c_data["cameras"])

        # Validate all indices are in range
        out_of_range = {i for i in drop_indices if i < 1 or i > n_views}
        if out_of_range:
            raise ValueError(
                f"View indices out of range (dataset has {n_views} views): "
                f"{sorted(out_of_range)}"
            )

        # Build keep mask (convert 1-based drop indices to 0-based)
        keep = [i for i in range(n_views) if (i + 1) not in drop_indices]

        if not keep:
            raise ValueError(
                f"Cannot drop all {n_views} views — at least one view must remain"
            )

        n_dropped = n_views - len(keep)
        logger.info(
            f"[Body2COLMAP] DropViews: dropping {n_dropped} of {n_views} views, "
            f"{len(keep)} remaining"
        )

        # Filter tensors
        keep_tensor = torch.tensor(keep, dtype=torch.long)
        filtered_images = images[keep_tensor]
        filtered_masks = masks[keep_tensor]

        # Filter metadata lists
        filtered_b2c_data = dict(b2c_data)
        filtered_b2c_data["cameras"] = [b2c_data["cameras"][i] for i in keep]
        filtered_b2c_data["image_names"] = [b2c_data["image_names"][i] for i in keep]

        return (filtered_b2c_data, filtered_images, filtered_masks)
