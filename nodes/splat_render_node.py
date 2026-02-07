"""Splat render node - renders Gaussian Splats using gsplat."""

import logging
import time

import comfy.utils
from body2colmap.path import OrbitPath
from body2colmap.camera import Camera
from body2colmap.utils import compute_default_focal_length, compute_auto_orbit_radius
from ..core.comfy_utils import rendered_to_comfy
from ..core.camera_utils import focal_length_mm_to_pixels

logger = logging.getLogger(__name__)


class Body2COLMAP_RenderSplat:
    """Render Gaussian Splats from camera path configuration."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "MASK", "B2C_COLMAP_METADATA")
    RETURN_NAMES = ("images", "masks", "b2c_data")
    OUTPUT_TOOLTIPS = (
        "Batch of rendered RGB images (connect to SaveImage or PreviewImage)",
        "Batch of alpha masks for each image",
        "Body2COLMAP dataset metadata (connect to ExportCOLMAP or SaveDataset)"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "splat_scene": ("SPLAT_SCENE",),
                "width": ("INT", {
                    "default": 720,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Image height in pixels"
                }),
            },
            "optional": {
                # Camera path configuration (optional if b2c_data provides cameras)
                "path_config": ("B2C_PATH_CONFIG", {
                    "tooltip": "Camera path from path generator. If not connected, cameras from b2c_data are reused."
                }),

                # Metadata from mesh renderer (for consistent framing or camera reuse)
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Metadata from a previous render. Used for framing bounds, or to reuse exact camera positions when path_config is not connected."
                }),

                # Camera parameters
                "focal_length_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Focal length in mm, 35mm full-frame equivalent (0=auto ~43mm, 50mm=standard)"
                }),
                "fill_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of viewport should contain scene (for auto-radius)"
                }),

                # Background color (no mesh color for splats - they have baked appearance)
                "bg_color_r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color red channel"
                }),
                "bg_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color green channel"
                }),
                "bg_color_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color blue channel"
                }),

                # Device selection for torch/gsplat
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for rendering (cuda strongly recommended for speed)"
                }),

                # Point cloud sampling for COLMAP export
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from Gaussian centers for COLMAP initialization"
                }),
                "override_pointcloud": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate new point cloud from splat (if False, preserves original from b2c_data if available)"
                }),
            }
        }

    def render(self, splat_scene, width, height,
               path_config=None, b2c_data=None,
               focal_length_mm=0.0, fill_ratio=0.8,
               bg_color_r=1.0, bg_color_g=1.0, bg_color_b=1.0,
               device="cuda",
               pointcloud_samples=10000,
               override_pointcloud=False):
        """
        Render all camera positions and return batch of images + masks.

        Args:
            splat_scene: SplatScene object from LoadSplat node
            width: Image width in pixels
            height: Image height in pixels
            path_config: B2C_PATH_CONFIG from path generator node (optional if b2c_data has cameras)
            b2c_data: Optional metadata; used for framing bounds or camera reuse
            focal_length_mm: Focal length in mm (0=auto)
            fill_ratio: Viewport fill ratio for auto-radius
            bg_color_r/g/b: Background color RGB components
            device: torch device ("cuda" or "cpu")

        Returns:
            images: Tensor of shape [N, H, W, 3] in [0,1] range (ComfyUI IMAGE format)
            masks: Tensor of shape [N, H, W] in [0,1] range (alpha channel)
            b2c_data: B2C_COLMAP_METADATA with cameras, point cloud, and image names
        """
        # Import SplatRenderer
        try:
            from body2colmap.splat_renderer import SplatRenderer
        except ImportError as e:
            raise ImportError(
                "Failed to import body2colmap.splat_renderer. "
                "Make sure body2colmap is installed with splat support: "
                "pip install body2colmap[splat]"
            ) from e

        # Validate inputs: need either path_config or cameras in b2c_data
        if path_config is None and (b2c_data is None or "cameras" not in b2c_data):
            raise ValueError(
                "Either path_config or b2c_data (with cameras) must be provided. "
                "Connect a path generator node, or connect b2c_data from a previous render "
                "to reuse its camera positions."
            )

        logger.info(
            f"[Body2COLMAP] Rendering splat scene: "
            f"{len(splat_scene)} Gaussians, SH degree {splat_scene.sh_degree}"
        )

        if path_config is not None:
            # Generate cameras from path configuration
            cameras = self._cameras_from_path(
                path_config, b2c_data, splat_scene,
                width, height, focal_length_mm, fill_ratio
            )
        else:
            # Reuse cameras from b2c_data
            cameras = b2c_data["cameras"]
            logger.info(
                f"[Body2COLMAP] Reusing {len(cameras)} cameras from b2c_data"
            )

        # Prepare background color
        bg_color = (bg_color_r, bg_color_g, bg_color_b)

        # Create SplatRenderer
        logger.info(
            f"[Body2COLMAP] Creating SplatRenderer "
            f"(size={width}x{height}, device={device})..."
        )
        t0 = time.time()
        renderer = SplatRenderer(
            scene=splat_scene,
            render_size=(width, height),
            device=device
        )
        logger.info(f"[Body2COLMAP] Renderer created ({time.time() - t0:.2f}s)")

        # Render all frames
        rendered_images = []
        n_frames = len(cameras)
        logger.info(f"[Body2COLMAP] Starting render loop: {n_frames} frames")
        pbar = comfy.utils.ProgressBar(n_frames)

        for i, camera in enumerate(cameras):
            frame_start = time.time()
            if i == 0:
                logger.info("[Body2COLMAP] Rendering first frame...")

            # Render splat (returns RGBA uint8)
            img = renderer.render(camera=camera, bg_color=bg_color)

            if i == 0:
                logger.info(
                    f"[Body2COLMAP] First frame complete ({time.time() - frame_start:.2f}s)"
                )

            rendered_images.append(img)
            frame_time = time.time() - frame_start
            logger.debug(
                f"[Body2COLMAP] Frame {i+1}/{n_frames} rendered ({frame_time:.2f}s)"
            )
            pbar.update(1)

        logger.info("[Body2COLMAP] Render loop complete")

        # Convert to ComfyUI IMAGE and MASK formats
        logger.info("[Body2COLMAP] Converting rendered images to ComfyUI format...")
        t0 = time.time()
        images_tensor, masks_tensor = rendered_to_comfy(rendered_images)
        logger.info(f"[Body2COLMAP] Conversion complete ({time.time() - t0:.2f}s)")

        # Determine point cloud to use
        if override_pointcloud or not b2c_data or "points_3d" not in b2c_data:
            # Generate new point cloud from splat scene
            if override_pointcloud:
                logger.info(f"[Body2COLMAP] Generating new point cloud (override_pointcloud=True)")
            else:
                logger.info(f"[Body2COLMAP] No point cloud in metadata, generating from splat scene")

            logger.info(f"[Body2COLMAP] Sampling {pointcloud_samples} points from splat scene...")
            t0 = time.time()
            points, colors = splat_scene.get_point_cloud(n_samples=pointcloud_samples)
            logger.info(f"[Body2COLMAP] Point cloud sampled ({time.time() - t0:.2f}s)")
        else:
            # Preserve original point cloud from metadata
            points, colors = b2c_data["points_3d"]
            logger.info(
                f"[Body2COLMAP] Using original point cloud from metadata "
                f"({len(points)} points)"
            )

        # Generate standardized filenames (1-based indexing with trailing underscore)
        image_names = [f"frame_{i+1:05d}_.png" for i in range(len(cameras))]

        # Package metadata for serialization (no scene object - not serializable)
        b2c_output = {
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": (points, colors),
            "resolution": (width, height),
        }

        # Pass through framing bounds if they were provided in input metadata
        if b2c_data and "framing_bounds" in b2c_data:
            b2c_output["framing_bounds"] = b2c_data["framing_bounds"]

        return (images_tensor, masks_tensor, b2c_output)

    def _cameras_from_path(self, path_config, b2c_data, splat_scene,
                           width, height, focal_length_mm, fill_ratio):
        """Generate cameras from a path configuration."""
        # Determine focal length in pixels
        if focal_length_mm <= 0:
            focal_length = compute_default_focal_length(width)
        else:
            focal_length = focal_length_mm_to_pixels(focal_length_mm, width)

        # Get framing preset from path config
        framing = path_config.get("framing", "full")
        pattern = path_config["pattern"]
        params = path_config["params"].copy()  # Don't modify original

        # Try to use framing bounds from metadata (computed by mesh renderer)
        bounds = None
        if b2c_data and "framing_bounds" in b2c_data:
            framing_bounds_dict = b2c_data["framing_bounds"]
            if framing in framing_bounds_dict:
                bounds = framing_bounds_dict[framing]
                logger.info(f"[Body2COLMAP] Using '{framing}' framing bounds from metadata")
            else:
                available = list(framing_bounds_dict.keys())
                logger.warning(
                    f"[Body2COLMAP] Framing preset '{framing}' not in metadata. "
                    f"Available presets: {available}. Falling back to splat scene bounds."
                )

        if bounds is None:
            if framing != "full" and b2c_data is None:
                logger.warning(
                    f"[Body2COLMAP] Framing preset '{framing}' requested but no metadata provided. "
                    "Using full splat scene bounds. Connect mesh renderer metadata to use framing."
                )
            bounds = splat_scene.get_bounds()

        # Compute orbit center from bounds
        orbit_center = (bounds[0] + bounds[1]) / 2.0

        # Auto-compute radius if not specified in path config
        if params.get("radius") is None:
            params["radius"] = compute_auto_orbit_radius(
                bounds=bounds,
                render_size=(width, height),
                focal_length=focal_length,
                fill_ratio=fill_ratio
            )

        # Create camera template
        camera_template = Camera(
            focal_length=(focal_length, focal_length),
            image_size=(width, height)
        )

        # Create OrbitPath and generate cameras based on pattern
        logger.info(
            f"[Body2COLMAP] Creating camera path: {pattern} "
            f"with radius={params['radius']:.3f}"
        )
        t0 = time.time()
        path_gen = OrbitPath(target=orbit_center, radius=params["radius"])

        if pattern == "circular":
            cameras = path_gen.circular(
                n_frames=params["n_frames"],
                elevation_deg=params["elevation_deg"],
                start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
                camera_template=camera_template
            )
        elif pattern == "sinusoidal":
            cameras = path_gen.sinusoidal(
                n_frames=params["n_frames"],
                amplitude_deg=params["amplitude_deg"],
                n_cycles=params["n_cycles"],
                start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
                camera_template=camera_template
            )
        elif pattern == "helical":
            cameras = path_gen.helical(
                n_frames=params["n_frames"],
                n_loops=params["n_loops"],
                amplitude_deg=params["amplitude_deg"],
                lead_in_deg=params.get("lead_in_deg", 45.0),
                lead_out_deg=params.get("lead_out_deg", 45.0),
                start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
                camera_template=camera_template
            )
        else:
            raise ValueError(f"Unknown path pattern: {pattern}")

        logger.info(
            f"[Body2COLMAP] Camera path created: {len(cameras)} cameras "
            f"({time.time() - t0:.2f}s)"
        )
        return cameras
