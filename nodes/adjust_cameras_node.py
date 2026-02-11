"""Adjust camera positions in an existing dataset without affecting images."""

import logging

import numpy as np
from body2colmap.camera import Camera
from body2colmap.coordinates import spherical_to_cartesian

logger = logging.getLogger(__name__)


class Body2COLMAP_AdjustCameras:
    """Shift camera elevation (spherical) and/or height (cylindrical) in a dataset.

    Useful for perspective shifting when rendering splats: e.g. adding 5° elevation
    to a circular-path dataset at 0° produces the same camera placements as generating
    a circular path at 5° elevation.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "adjust"
    RETURN_TYPES = ("B2C_COLMAP_METADATA",)
    RETURN_NAMES = ("b2c_data",)
    OUTPUT_TOOLTIPS = (
        "Dataset metadata with adjusted camera positions (images unchanged)",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset whose cameras will be adjusted"
                }),
            },
            "optional": {
                "elevation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -89.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": (
                        "Elevation offset in degrees (spherical motion). "
                        "Moves cameras along the orbit sphere, like changing "
                        "elevation_deg in a path node."
                    )
                }),
                "height": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Vertical offset in meters (cylindrical motion). "
                        "Translates cameras up/down while keeping horizontal "
                        "distance constant."
                    )
                }),
            }
        }

    def adjust(self, b2c_data, elevation_deg=0.0, height=0.0):
        """Adjust camera positions by elevation and/or height offset."""
        if elevation_deg == 0.0 and height == 0.0:
            logger.info("[Body2COLMAP] AdjustCameras: no offsets, passing through unchanged")
            return (b2c_data,)

        cameras = b2c_data["cameras"]
        n = len(cameras)

        # Estimate orbit center from camera positions
        positions = np.array([c.position for c in cameras], dtype=np.float64)
        orbit_center = positions.mean(axis=0)

        logger.info(
            f"[Body2COLMAP] AdjustCameras: {n} cameras, "
            f"elevation_offset={elevation_deg}°, height_offset={height}m, "
            f"orbit_center=[{orbit_center[0]:.3f}, {orbit_center[1]:.3f}, {orbit_center[2]:.3f}]"
        )

        new_cameras = []
        for cam in cameras:
            v = cam.position.astype(np.float64) - orbit_center
            r = np.linalg.norm(v)

            # Cartesian to spherical (Y-up, azimuth 0° = +Z)
            azimuth_deg = float(np.degrees(np.arctan2(v[0], v[2])))
            elevation = float(np.degrees(np.arcsin(np.clip(v[1] / r, -1.0, 1.0))))

            # Apply elevation offset (clamp to avoid pole singularities)
            new_elevation = float(np.clip(elevation + elevation_deg, -89.9, 89.9))

            # Spherical back to Cartesian, then apply height offset
            new_pos = orbit_center + spherical_to_cartesian(r, azimuth_deg, new_elevation)
            new_pos[1] += height

            new_cam = Camera(
                focal_length=(cam.fx, cam.fy),
                image_size=(cam.width, cam.height),
                principal_point=(cam.cx, cam.cy),
                position=new_pos.astype(np.float32),
                rotation=cam.rotation.copy(),
            )
            new_cam.look_at(orbit_center.astype(np.float32))
            new_cameras.append(new_cam)

        result = dict(b2c_data)
        result["cameras"] = new_cameras
        return (result,)
