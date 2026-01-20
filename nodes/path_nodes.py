"""Camera path configuration nodes for Body2COLMAP."""


class Body2COLMAP_CircularPath:
    """Configure circular camera orbit path."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "configure"
    RETURN_TYPES = ("B2C_PATH_CONFIG",)
    RETURN_NAMES = ("path_config",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_frames": ("INT", {
                    "default": 81,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of camera positions around the orbit"
                }),
                "elevation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -89.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Camera elevation angle (0=eye level, positive=above)"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius in meters (0=auto-compute from mesh bounds)"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "tooltip": "Starting azimuth angle (0=front)"
                }),
            }
        }

    def configure(self, n_frames, elevation_deg, radius=0.0, start_azimuth_deg=0.0):
        """Return circular path configuration."""
        return ({
            "pattern": "circular",
            "params": {
                "n_frames": int(n_frames),
                "elevation_deg": float(elevation_deg),
                "radius": float(radius) if radius > 0 else None,
                "start_azimuth_deg": float(start_azimuth_deg),
            }
        },)


class Body2COLMAP_SinusoidalPath:
    """Configure sinusoidal camera orbit with oscillating elevation."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "configure"
    RETURN_TYPES = ("B2C_PATH_CONFIG",)
    RETURN_NAMES = ("path_config",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_frames": ("INT", {
                    "default": 81,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of camera positions"
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Maximum elevation deviation from center"
                }),
                "n_cycles": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of up/down oscillations per full rotation"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius in meters (0=auto-compute)"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "tooltip": "Starting azimuth angle"
                }),
            }
        }

    def configure(self, n_frames, amplitude_deg, n_cycles,
                  radius=0.0, start_azimuth_deg=0.0):
        """Return sinusoidal path configuration."""
        return ({
            "pattern": "sinusoidal",
            "params": {
                "n_frames": int(n_frames),
                "amplitude_deg": float(amplitude_deg),
                "n_cycles": int(n_cycles),
                "radius": float(radius) if radius > 0 else None,
                "start_azimuth_deg": float(start_azimuth_deg),
            }
        },)


class Body2COLMAP_HelicalPath:
    """Configure helical camera path - best for 3D Gaussian Splatting training."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "configure"
    RETURN_TYPES = ("B2C_PATH_CONFIG",)
    RETURN_NAMES = ("path_config",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_frames": ("INT", {
                    "default": 81,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total number of frames across all loops"
                }),
                "n_loops": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of full 360° rotations"
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Elevation range: camera goes from -amplitude to +amplitude"
                }),
            },
            "optional": {
                "lead_in_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at bottom before ascending"
                }),
                "lead_out_deg": ("FLOAT", {
                    "default": 90.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at top after ascending"
                }),
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius (0=auto-compute)"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                }),
            }
        }

    def configure(self, n_frames, n_loops, amplitude_deg,
                  lead_in_deg=30.0, lead_out_deg=90.0,
                  radius=0.0, start_azimuth_deg=0.0):
        """Return helical path configuration."""
        return ({
            "pattern": "helical",
            "params": {
                "n_frames": int(n_frames),
                "n_loops": int(n_loops),
                "amplitude_deg": float(amplitude_deg),
                "lead_in_deg": float(lead_in_deg),
                "lead_out_deg": float(lead_out_deg),
                "radius": float(radius) if radius > 0 else None,
                "start_azimuth_deg": float(start_azimuth_deg),
            }
        },)
