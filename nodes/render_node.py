"""Render node for Body2COLMAP - generates multi-view images."""

import logging
import time

import comfy.utils
from body2colmap.renderer import Renderer
from body2colmap.path import OrbitPath
from body2colmap.camera import Camera
from body2colmap.utils import compute_default_focal_length, compute_auto_orbit_radius
from ..core.sam3d_adapter import sam3d_output_to_scene
from ..core.comfy_utils import rendered_to_comfy
from ..core.camera_utils import focal_length_mm_to_pixels

logger = logging.getLogger(__name__)


class Body2COLMAP_Render:
    """Render multi-view images of mesh from camera path configuration."""

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
                "mesh_data": ("SAM3D_OUTPUT",),
                "path_config": ("B2C_PATH_CONFIG",),
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
                "render_mode": ([
                    "mesh",
                    "depth",
                    "skeleton",
                    "mesh+skeleton",
                    "depth+skeleton"
                ], {
                    "default": "depth+skeleton",
                    "tooltip": "What to render: mesh surface, depth map, skeleton, or composites"
                }),
            },
            "optional": {
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
                    "tooltip": "How much of viewport should contain mesh (for auto-radius)"
                }),

                # Mesh rendering options
                "mesh_color_r": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color red channel"
                }),
                "mesh_color_g": ("FLOAT", {
                    "default": 0.74,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color green channel"
                }),
                "mesh_color_b": ("FLOAT", {
                    "default": 0.86,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color blue channel"
                }),
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

                # Skeleton rendering options (for skeleton modes)
                "skeleton_format": ([
                    "openpose_body25_hands",
                    "mhr70"
                ], {"default": "openpose_body25_hands"}),
                "joint_radius": ("FLOAT", {
                    "default": 0.006,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Sphere radius for skeleton joints (meters)"
                }),
                "bone_radius": ("FLOAT", {
                    "default": 0.003,
                    "min": 0.001,
                    "max": 0.05,
                    "step": 0.001,
                    "tooltip": "Cylinder radius for skeleton bones (meters)"
                }),

                # Depth rendering options
                "depth_colormap": ([
                    "grayscale",
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma"
                ], {"default": "grayscale"}),

                # Point cloud sampling for COLMAP export
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh for COLMAP initialization"
                }),
            }
        }

    def render(self, mesh_data, path_config, width, height, render_mode,
               focal_length_mm=0.0, fill_ratio=0.8,
               mesh_color_r=0.65, mesh_color_g=0.74, mesh_color_b=0.86,
               bg_color_r=1.0, bg_color_g=1.0, bg_color_b=1.0,
               skeleton_format="openpose_body25_hands",
               joint_radius=0.006, bone_radius=0.003,
               depth_colormap="grayscale",
               pointcloud_samples=10000):
        """
        Render all camera positions and return batch of images + masks.

        Returns:
            images: Tensor of shape [N, H, W, 3] in [0,1] range (ComfyUI IMAGE format)
            masks: Tensor of shape [N, H, W] in [0,1] range (alpha channel)
            b2c_data: B2C_COLMAP_METADATA with cameras, point cloud, and image names
        """
        # Get framing preset from path config
        framing = path_config.get("framing", "full")

        # Convert SAM3D output to Scene
        # Always load skeleton if available to compute all framing bounds for metadata
        include_skeleton = True
        logger.info("[Body2COLMAP] Converting SAM3D output to scene...")
        t0 = time.time()
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=include_skeleton)
        logger.info(f"[Body2COLMAP] Scene conversion complete ({time.time() - t0:.2f}s)")

        # Determine focal length in pixels
        # Convert from mm (35mm full-frame equivalent) to pixels
        if focal_length_mm <= 0:
            # Auto-compute default (~43mm equivalent, ~47° FOV)
            focal_length = compute_default_focal_length(width)
        else:
            focal_length = focal_length_mm_to_pixels(focal_length_mm, width)

        # Compute ALL framing bounds for metadata (allows splat renderer to choose later)
        logger.info("[Body2COLMAP] Computing framing bounds for all presets...")
        all_framing_bounds = {}

        # Always compute full bounds
        all_framing_bounds["full"] = scene.get_bounds()

        # Compute partial framing bounds if skeleton is available
        if scene.skeleton_joints is not None:
            for preset in ["torso", "bust", "head"]:
                try:
                    all_framing_bounds[preset] = scene.get_framing_bounds(preset=preset)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"[Body2COLMAP] Could not compute {preset} framing bounds: {e}")
        else:
            logger.info("[Body2COLMAP] No skeleton data - only 'full' framing available")

        # Get bounds for the selected framing preset
        pattern = path_config["pattern"]
        params = path_config["params"].copy()  # Don't modify original

        if framing in all_framing_bounds:
            if framing != "full":
                logger.info(f"[Body2COLMAP] Using framing preset: {framing}")
            current_bounds = all_framing_bounds[framing]
        else:
            logger.warning(
                f"[Body2COLMAP] Framing preset '{framing}' not available, falling back to 'full'"
            )
            current_bounds = all_framing_bounds["full"]

        # Compute orbit center from selected framing bounds
        orbit_center = (current_bounds[0] + current_bounds[1]) / 2.0

        # Auto-compute radius if not specified in path config
        if params.get("radius") is None:
            params["radius"] = compute_auto_orbit_radius(
                bounds=current_bounds,
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
        logger.info(f"[Body2COLMAP] Creating camera path: {pattern} with radius={params['radius']:.3f}")
        t0 = time.time()
        path_gen = OrbitPath(target=orbit_center, radius=params["radius"])

        if pattern == "circular":
            cameras = path_gen.circular(
                n_frames=params["n_frames"],
                elevation_deg=params["elevation_deg"],
                start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
                overlap=params.get("overlap", 1),
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
        logger.info(f"[Body2COLMAP] Camera path created: {len(cameras)} cameras ({time.time() - t0:.2f}s)")

        # Prepare render colors
        mesh_color = (mesh_color_r, mesh_color_g, mesh_color_b)
        bg_color = (bg_color_r, bg_color_g, bg_color_b)

        # Map "grayscale" to None (no colormap = grayscale depth)
        depth_cmap = None if depth_colormap == "grayscale" else depth_colormap

        # Create renderer - requires scene and render_size tuple
        logger.info(f"[Body2COLMAP] Creating renderer (size={width}x{height})...")
        t0 = time.time()
        renderer = Renderer(scene=scene, render_size=(width, height))
        logger.info(f"[Body2COLMAP] Renderer created ({time.time() - t0:.2f}s)")

        # Render all frames
        rendered_images = []
        n_frames = len(cameras)
        logger.info(f"[Body2COLMAP] Starting render loop: {n_frames} frames, mode={render_mode}")
        pbar = comfy.utils.ProgressBar(n_frames)

        for i, camera in enumerate(cameras):
            frame_start = time.time()
            if i == 0:
                logger.info(f"[Body2COLMAP] Rendering first frame...")
            # Determine render mode
            if render_mode == "mesh":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_mesh...")
                img = renderer.render_mesh(
                    camera=camera,
                    mesh_color=mesh_color,
                    bg_color=bg_color,
                )
            elif render_mode == "depth":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_depth...")
                img = renderer.render_depth(
                    camera=camera,
                    colormap=depth_cmap,
                )
            elif render_mode == "skeleton":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_skeleton...")
                img = renderer.render_skeleton(
                    camera=camera,
                    target_format=skeleton_format,
                    joint_radius=joint_radius,
                    bone_radius=bone_radius,
                )
            elif render_mode == "mesh+skeleton":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_composite (mesh+skeleton)...")
                img = renderer.render_composite(
                    camera=camera,
                    modes={
                        "mesh": {"color": mesh_color, "bg_color": bg_color},
                        "skeleton": {
                            "target_format": skeleton_format,
                            "joint_radius": joint_radius,
                            "bone_radius": bone_radius
                        }
                    }
                )
            elif render_mode == "depth+skeleton":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_composite (depth+skeleton)...")
                img = renderer.render_composite(
                    camera=camera,
                    modes={
                        "depth": {"colormap": depth_cmap},
                        "skeleton": {
                            "target_format": skeleton_format,
                            "joint_radius": joint_radius,
                            "bone_radius": bone_radius
                        }
                    }
                )
            else:
                raise ValueError(f"Unknown render mode: {render_mode}")
            if i == 0:
                logger.info(f"[Body2COLMAP] First frame complete ({time.time() - frame_start:.2f}s)")

            rendered_images.append(img)
            frame_time = time.time() - frame_start
            logger.debug(f"[Body2COLMAP] Frame {i+1}/{n_frames} rendered ({frame_time:.2f}s)")
            pbar.update(1)

        logger.info(f"[Body2COLMAP] Render loop complete")

        # Convert to ComfyUI IMAGE and MASK formats
        logger.info("[Body2COLMAP] Converting rendered images to ComfyUI format...")
        t0 = time.time()
        images_tensor, masks_tensor = rendered_to_comfy(rendered_images)
        logger.info(f"[Body2COLMAP] Conversion complete ({time.time() - t0:.2f}s)")

        # Sample point cloud from scene (do this while we still have the scene!)
        logger.info(f"[Body2COLMAP] Sampling {pointcloud_samples} points from scene...")
        t0 = time.time()
        points, colors = scene.get_point_cloud(n_samples=pointcloud_samples)
        logger.info(f"[Body2COLMAP] Point cloud sampled ({time.time() - t0:.2f}s)")

        # Generate standardized filenames (1-based indexing with trailing underscore)
        image_names = [f"frame_{i+1:05d}_.png" for i in range(len(cameras))]

        # Package metadata for serialization (no scene object - not serializable)
        b2c_data = {
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": (points, colors),
            "resolution": (width, height),
            "framing_bounds": all_framing_bounds,  # Dict of all computed framing bounds
        }

        return (images_tensor, masks_tensor, b2c_data)
