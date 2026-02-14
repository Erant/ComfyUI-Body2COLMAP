"""Rotate Views node - set the starting azimuth of the image sequence."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _normalize_angle(deg: float) -> float:
    """Wrap an angle in degrees to the range (-180, 180]."""
    return ((deg + 180.0) % 360.0) - 180.0


class Body2COLMAP_RotateViews:
    """Set the starting azimuth of the view sequence.

    Cyclically reorders the cameras and images so that ``frame_00001``
    corresponds to the view closest to the given azimuth.  The camera–image
    pairing is preserved and image names are re-numbered sequentially.

    Because the parameter is an absolute azimuth (not a relative offset),
    applying the node multiple times with the same value is idempotent.

    Azimuth is relative to the skeleton's front:
      - 0° = front
      - +90° = right side
      - -90° = left side
      - ±180° = back
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
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": (
                        "Absolute azimuth for frame_00001, relative to the "
                        "skeleton's front. 0 = front, 90 = right, "
                        "-90 = left, ±180 = back."
                    ),
                }),
            },
        }

    def rotate(self, b2c_data, images, masks, start_azimuth_deg):
        cameras = b2c_data["cameras"]
        n_views = len(cameras)

        if n_views == 0:
            raise ValueError("Cannot rotate an empty dataset")

        orbit_target = b2c_data.get("orbit_target")
        if orbit_target is None:
            raise ValueError(
                "b2c_data is missing 'orbit_target'. Rotate Views requires "
                "data from a Render node (not supported with merged datasets)."
            )
        forward_azimuth_deg = b2c_data.get("forward_azimuth_deg", 0.0)

        # Compute each camera's azimuth relative to skeleton front
        azimuths = []
        for cam in cameras:
            dx = cam.position[0] - orbit_target[0]
            dz = cam.position[2] - orbit_target[2]
            orbit_az = np.degrees(np.arctan2(dx, dz))
            rel_az = _normalize_angle(orbit_az - forward_azimuth_deg)
            azimuths.append(rel_az)

        # Find the view closest to the target azimuth
        target = _normalize_angle(start_azimuth_deg)
        diffs = [abs(_normalize_angle(az - target)) for az in azimuths]
        best_idx = int(np.argmin(diffs))

        if best_idx == 0:
            logger.info(
                "[Body2COLMAP] RotateViews: start_azimuth=%.1f° → view 1 "
                "(azimuth %.1f°) is already first, no change",
                start_azimuth_deg, azimuths[0]
            )
            return (b2c_data, images, masks)

        logger.info(
            "[Body2COLMAP] RotateViews: start_azimuth=%.1f° → view %d "
            "(azimuth %.1f°) becomes frame_00001",
            start_azimuth_deg, best_idx + 1, azimuths[best_idx]
        )

        # Build new order starting from best_idx
        order = [(best_idx + i) % n_views for i in range(n_views)]

        # Reorder tensors
        order_tensor = torch.tensor(order, dtype=torch.long)
        rotated_images = images[order_tensor]
        rotated_masks = masks[order_tensor]

        # Reorder metadata (cameras move with their images)
        rotated_b2c_data = dict(b2c_data)
        rotated_b2c_data["cameras"] = [cameras[i] for i in order]
        rotated_b2c_data["image_names"] = [
            f"frame_{j+1:05d}_.png" for j in range(n_views)
        ]

        return (rotated_b2c_data, rotated_images, rotated_masks)
