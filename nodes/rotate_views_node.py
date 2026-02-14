"""Rotate Views node - cyclically shift the image sequence by a degree offset."""

import logging

import torch

logger = logging.getLogger(__name__)


class Body2COLMAP_RotateViews:
    """Cyclically rotate the view sequence by a given number of degrees.

    This shifts which image is assigned to ``frame_00001`` without altering
    the camera positions themselves.  The camera–image pairing is preserved
    (cameras and images rotate together), so COLMAP correspondence stays
    correct.  The effect is that the sequence "starts" from a different
    point on the orbit.

    A positive ``rotation_deg`` rotates the starting point in the positive-
    azimuth direction (i.e. the images shift *backward* by that many
    positions).

    Example:
        With 36 frames (10° spacing) and ``rotation_deg = 90``, what was
        previously view 10 becomes the new ``frame_00001``.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "rotate"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Dataset metadata with the view sequence rotated",
        "Image batch with the view sequence rotated",
        "Mask batch with the view sequence rotated",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset metadata from a Render node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image batch to rotate"
                }),
                "masks": ("MASK", {
                    "tooltip": "Mask batch to rotate"
                }),
                "rotation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": (
                        "Degrees to rotate the sequence. With 36 frames, "
                        "90° shifts the start point by ~10 frames."
                    ),
                }),
            },
        }

    def rotate(self, b2c_data, images, masks, rotation_deg):
        n_views = len(b2c_data["cameras"])

        if n_views == 0:
            raise ValueError("Cannot rotate an empty dataset")

        # Convert degrees to a frame shift.  Assume views are evenly spaced
        # over a full 360° orbit.
        shift = round(rotation_deg * n_views / 360.0) % n_views

        if shift == 0:
            logger.info(
                "[Body2COLMAP] RotateViews: rotation_deg=%.1f results in "
                "zero shift, passing through unchanged", rotation_deg
            )
            return (b2c_data, images, masks)

        logger.info(
            "[Body2COLMAP] RotateViews: rotation_deg=%.1f → shifting sequence "
            "by %d of %d views", rotation_deg, shift, n_views
        )

        # Build the new index order: [shift, shift+1, ..., N-1, 0, 1, ..., shift-1]
        order = [(shift + i) % n_views for i in range(n_views)]

        # Rotate tensors
        order_tensor = torch.tensor(order, dtype=torch.long)
        rotated_images = images[order_tensor]
        rotated_masks = masks[order_tensor]

        # Rotate metadata lists (cameras move with their images)
        rotated_b2c_data = dict(b2c_data)
        rotated_b2c_data["cameras"] = [b2c_data["cameras"][i] for i in order]
        # Re-number image names sequentially so frame_00001 is always first
        rotated_b2c_data["image_names"] = [
            f"frame_{j+1:05d}_.png" for j in range(n_views)
        ]

        return (rotated_b2c_data, rotated_images, rotated_masks)
