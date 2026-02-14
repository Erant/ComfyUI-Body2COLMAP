"""Replace Views node - swap matching camera views between two datasets."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _camera_positions(cameras):
    """Extract an (N, 3) array of camera positions."""
    return np.stack([cam.position for cam in cameras], axis=0)


def _scene_scale(positions):
    """Characteristic scale = diagonal of bounding box of camera positions."""
    if len(positions) < 2:
        return 1.0
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return float(diag) if diag > 0 else 1.0


class Body2COLMAP_ReplaceViews:
    """Replace views in a base dataset with matching views from a second dataset.

    For every camera in the base set, find the closest camera in the
    replacement set.  If the normalised position distance is below the
    tolerance threshold, the base view (camera, image, mask) is replaced
    by the replacement view.  The output is always the same size as the
    base set.  Multiple base cameras at the same location (e.g. from
    orbit overlap) will each find their closest replacement independently.

    Typical use-case: re-render a subset of views (e.g. with different
    settings or a trained splat) and merge them back into the original orbit.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "replace"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Base dataset with matched views replaced",
        "Image batch with replaced views",
        "Mask batch with replaced views",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Base dataset whose views may be replaced"
                }),
                "base_images": ("IMAGE", {
                    "tooltip": "Base image batch"
                }),
                "base_masks": ("MASK", {
                    "tooltip": "Base mask batch"
                }),
                "replacement_b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Replacement dataset with re-rendered views"
                }),
                "replacement_images": ("IMAGE", {
                    "tooltip": "Replacement image batch"
                }),
                "replacement_masks": ("MASK", {
                    "tooltip": "Replacement mask batch"
                }),
                "tolerance_pct": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Match tolerance as a percentage of the base camera "
                        "bounding-box diagonal.  Two cameras whose position "
                        "distance is below this fraction are considered the "
                        "same view.  0.1 = 0.1%."
                    )
                }),
            },
        }

    def replace(
        self,
        base_b2c_data,
        base_images,
        base_masks,
        replacement_b2c_data,
        replacement_images,
        replacement_masks,
        tolerance_pct,
    ):
        base_cameras = base_b2c_data["cameras"]
        repl_cameras = replacement_b2c_data["cameras"]

        base_pos = _camera_positions(base_cameras)   # (N, 3)
        repl_pos = _camera_positions(repl_cameras)    # (M, 3)

        scale = _scene_scale(base_pos)
        threshold = (tolerance_pct / 100.0) * scale

        # For each base camera, find the closest replacement camera.
        # This ensures every base view that has a nearby replacement gets
        # swapped, even when multiple base cameras share a location (overlap).
        matches = {}  # base_idx -> (repl_idx, distance)
        for b_idx in range(len(base_cameras)):
            dists = np.linalg.norm(repl_pos - base_pos[b_idx], axis=1)
            r_idx = int(np.argmin(dists))
            d = float(dists[r_idx])
            if d <= threshold:
                matches[b_idx] = (r_idx, d)

        if not matches:
            logger.warning(
                "[Body2COLMAP] ReplaceViews: no cameras matched within "
                f"tolerance {tolerance_pct}% (threshold={threshold:.6f}, "
                f"scale={scale:.6f}). Returning base dataset unchanged."
            )
            return (base_b2c_data, base_images, base_masks)

        logger.info(
            f"[Body2COLMAP] ReplaceViews: replacing {len(matches)}/{len(base_cameras)} "
            f"views (tolerance={tolerance_pct}%, threshold={threshold:.6f})"
        )
        for b_idx, (r_idx, d) in sorted(matches.items()):
            logger.info(
                f"  base[{b_idx}] <- replacement[{r_idx}]  "
                f"(distance={d:.8f}, {d / scale * 100:.4f}%)"
            )

        # Build output tensors by cloning base and overwriting matched views
        out_images = base_images.clone()
        out_masks = base_masks.clone()

        out_cameras = list(base_cameras)  # shallow copy of list
        out_image_names = list(base_b2c_data["image_names"])

        for b_idx, (r_idx, _d) in matches.items():
            out_images[b_idx] = replacement_images[r_idx]
            out_masks[b_idx] = replacement_masks[r_idx]
            out_cameras[b_idx] = repl_cameras[r_idx]

        # Shallow copy metadata, update per-view lists
        out_b2c_data = dict(base_b2c_data)
        out_b2c_data["cameras"] = out_cameras
        out_b2c_data["image_names"] = out_image_names

        return (out_b2c_data, out_images, out_masks)
