"""Inject Anchor node - overwrite every frame sitting at the original camera."""

import logging

import numpy as np

from .replace_views_node import _camera_positions, _scene_scale

logger = logging.getLogger(__name__)


class Body2COLMAP_InjectAnchor:
    """Overwrite the orbit frame(s) at the original camera with a given image.

    When the Render node runs with ``override_cam_from_mesh``, one point of the
    orbit sits exactly at the SAM-3D-Body camera and ``b2c_data`` records that
    position as ``anchor_position``.  This node finds every frame at that
    position and replaces its pixels with ``anchor_image`` — typically the
    reference photo warped by Generate FirstLast, so a diffusion pass gets a
    real conditioning frame.

    The path passes through the anchor once, but more than one *frame* can land
    there: ``Circular Path`` with the default ``overlap=1`` duplicates the first
    camera as the last one.  The whole batch is therefore scanned; there is no
    early exit on the first match.

    Matching is by camera position, not by the recorded frame index, because
    Drop Views / Rotate Views / Filter FoV reorder and subset the camera list
    while carrying the rest of ``b2c_data`` through unchanged.

    With no ``anchor_image`` connected (or a dataset that carries no anchor
    frame), the inputs pass through untouched.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "inject"
    RETURN_TYPES = ("B2C_COLMAP_METADATA", "IMAGE", "MASK")
    RETURN_NAMES = ("b2c_data", "images", "masks")
    OUTPUT_TOOLTIPS = (
        "Metadata, unchanged (only pixels are swapped)",
        "Image batch with the anchor frame(s) replaced",
        "Mask batch with the anchor frame(s) made fully opaque",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset containing 'anchor_position' (Render node with override_cam_from_mesh)"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image batch to inject into"
                }),
                "masks": ("MASK", {
                    "tooltip": "Mask batch matching the images"
                }),
                "tolerance_pct": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Match tolerance as a percentage of the camera "
                        "bounding-box diagonal.  A frame whose distance to the "
                        "anchor is below this fraction is considered to sit on "
                        "the anchor.  0.1 = 0.1%."
                    )
                }),
            },
            "optional": {
                "anchor_image": ("IMAGE", {
                    "tooltip": (
                        "Anchor conditioning frame (e.g. from Generate FirstLast, "
                        "or Load Dataset's anchor_image output). "
                        "Leave unconnected to pass everything through unchanged."
                    )
                }),
            },
        }

    def inject(self, b2c_data, images, masks, tolerance_pct, anchor_image=None):
        """Replace every frame at the anchor position with ``anchor_image``."""
        # Nothing to inject: not wired up, or a dataset saved without an
        # anchor frame.  Pass through rather than failing the graph.
        if anchor_image is None or anchor_image.shape[0] == 0:
            logger.info(
                "[Body2COLMAP] InjectAnchor: no anchor image supplied, "
                "passing inputs through unchanged."
            )
            return (b2c_data, images, masks)

        if "anchor_position" not in b2c_data:
            raise ValueError(
                "b2c_data has no 'anchor_position', so there is no frame to "
                "inject into. Enable override_cam_from_mesh on the Render node "
                "(Circular or Helical Path), or disconnect anchor_image."
            )

        # Save → Load turns the stored ndarray into a plain list.
        anchor_position = np.asarray(b2c_data["anchor_position"], dtype=np.float32)

        positions = _camera_positions(b2c_data["cameras"])  # (N, 3)
        scale = _scene_scale(positions)
        threshold = (tolerance_pct / 100.0) * scale

        # Scan the full batch: an orbit can revisit the anchor (overlap=1 puts
        # it at the first *and* last frame), so every match must be replaced.
        distances = np.linalg.norm(positions - anchor_position, axis=1)
        matches = [int(i) for i in np.flatnonzero(distances <= threshold)]

        if not matches:
            logger.warning(
                "[Body2COLMAP] InjectAnchor: no frame matched the anchor within "
                f"tolerance {tolerance_pct}% (threshold={threshold:.6f}, "
                f"scale={scale:.6f}, closest={distances.min():.6f}). "
                "Returning inputs unchanged."
            )
            return (b2c_data, images, masks)

        anchor_frame = anchor_image[0]
        if tuple(anchor_frame.shape) != tuple(images.shape[1:]):
            raise ValueError(
                f"anchor_image shape {tuple(anchor_frame.shape)} does not match the "
                f"image batch frame shape {tuple(images.shape[1:])}. Render the anchor "
                f"at the same resolution as the orbit (Generate FirstLast uses the "
                f"render_size from the Render node)."
            )

        logger.info(
            f"[Body2COLMAP] InjectAnchor: injecting into {len(matches)}/{len(positions)} "
            f"frames (tolerance={tolerance_pct}%, threshold={threshold:.6f})"
        )
        for idx in matches:
            logger.info(f"  frame[{idx}] <- anchor  (distance={distances[idx]:.8f})")

        out_images = images.clone()
        out_masks = masks.clone()
        for idx in matches:
            out_images[idx] = anchor_frame
            # ComfyUI MASK here is inverted (1.0 = background, 0.0 = content),
            # so zeros marks the injected photo as fully opaque content.
            out_masks[idx] = 0.0

        # Pixels only — cameras and image_names keep their order and length.
        return (dict(b2c_data), out_images, out_masks)
