"""Filter FoV node - keep only views within an angular field of view."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _normalize_angle(deg: float) -> float:
    """Wrap an angle in degrees to the range (-180, 180]."""
    return ((deg + 180.0) % 360.0) - 180.0


def _compute_skeleton_relative_azimuths(cameras, orbit_target, forward_azimuth_deg):
    """Compute each camera's azimuth relative to the skeleton's front.

    Uses the OrbitPath convention: azimuth 0° = −Z direction.

    Args:
        cameras: List of Camera objects with ``.position`` attribute.
        orbit_target: np.ndarray shape (3,), the orbit center point.
        forward_azimuth_deg: Orbit azimuth that corresponds to the
            skeleton's front (0° for standard mode).

    Returns:
        List of floats in (-180, 180], where 0° = front, +90° = right,
        -90° (or 270°) = left, ±180° = back.
    """
    azimuths = []
    for cam in cameras:
        dx = cam.position[0] - orbit_target[0]
        dz = cam.position[2] - orbit_target[2]
        orbit_az = np.degrees(np.arctan2(dx, -dz))
        relative_az = _normalize_angle(orbit_az - forward_azimuth_deg)
        azimuths.append(relative_az)
    return azimuths


class Body2COLMAP_FilterFoV:
    """Keep only views whose azimuth falls within a field-of-view cone.

    Azimuth is measured relative to the skeleton's front:
      - 0° = directly in front
      - +90° = right side
      - -90° = left side
      - ±180° = directly behind

    Examples:
      azimuth=0, fov=180  → front hemisphere
      azimuth=180, fov=180 → back hemisphere
      azimuth=0, fov=90   → narrow frontal wedge (±45°)
      azimuth=90, fov=90  → right side wedge
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "filter"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Dataset metadata with only the views inside the FoV cone",
        "Filtered image batch",
        "Filtered mask batch",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset metadata from Render node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image batch to filter"
                }),
                "masks": ("MASK", {
                    "tooltip": "Mask batch to filter"
                }),
                "azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": (
                        "Center azimuth in degrees relative to skeleton front. "
                        "0 = front, 90 = right, -90 = left, ±180 = back."
                    )
                }),
                "fov_deg": ("FLOAT", {
                    "default": 180.0,
                    "min": 1.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": (
                        "Field of view width in degrees. Views within "
                        "±(fov/2) of the center azimuth are kept."
                    )
                }),
            },
        }

    def filter(self, b2c_data, images, masks, azimuth_deg, fov_deg):
        orbit_target = b2c_data.get("orbit_target")
        if orbit_target is None:
            raise ValueError(
                "b2c_data is missing 'orbit_target'. "
                "Re-run the Render node (requires updated Body2COLMAP)."
            )
        forward_azimuth_deg = b2c_data.get("forward_azimuth_deg", 0.0)

        cameras = b2c_data["cameras"]
        rel_azimuths = _compute_skeleton_relative_azimuths(
            cameras, orbit_target, forward_azimuth_deg
        )

        half_fov = fov_deg / 2.0
        center = _normalize_angle(azimuth_deg)

        keep = []
        for i, az in enumerate(rel_azimuths):
            diff = abs(_normalize_angle(az - center))
            if diff <= half_fov:
                keep.append(i)

        if not keep:
            raise ValueError(
                f"No views fall within azimuth={azimuth_deg}° ± {half_fov}°. "
                f"Camera azimuths range from {min(rel_azimuths):.1f}° to "
                f"{max(rel_azimuths):.1f}° (relative to skeleton front)."
            )

        n_total = len(cameras)
        logger.info(
            f"[Body2COLMAP] FilterFoV: azimuth={center:.1f}° fov={fov_deg:.1f}° "
            f"→ keeping {len(keep)}/{n_total} views"
        )

        # Filter tensors
        keep_tensor = torch.tensor(keep, dtype=torch.long)
        filtered_images = images[keep_tensor]
        filtered_masks = masks[keep_tensor]

        # Filter metadata (shallow copy, replace per-view lists)
        filtered_b2c_data = dict(b2c_data)
        filtered_b2c_data["cameras"] = [cameras[i] for i in keep]
        filtered_b2c_data["image_names"] = [b2c_data["image_names"][i] for i in keep]

        return (filtered_b2c_data, filtered_images, filtered_masks)
