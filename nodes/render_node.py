"""Render node for Body2COLMAP - generates multi-view images."""

import logging
import time

import numpy as np
import comfy.utils
from body2colmap.renderer import Renderer
from body2colmap.path import (
    OrbitPath,
    compute_helical_anchor_params,
    compute_original_camera_orbit_params,
)
from body2colmap.camera import Camera
from body2colmap.face import FaceLandmarkIngest
from body2colmap.utils import (
    compute_default_focal_length,
    compute_auto_orbit_radius,
    compute_original_view_framing,
)
from ..core.sam3d_adapter import sam3d_output_to_scene
from ..core.comfy_utils import rendered_to_comfy
from ..core.camera_utils import focal_length_mm_to_pixels, focal_length_pixels_to_mm

logger = logging.getLogger(__name__)


class Body2COLMAP_Render:
    """Render multi-view images of mesh from camera path configuration."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "MASK", "B2C_COLMAP_METADATA", "B2C_IMAGE_WARP")
    RETURN_NAMES = ("images", "masks", "b2c_data", "image_warp")
    OUTPUT_TOOLTIPS = (
        "Batch of rendered RGB images (connect to SaveImage or PreviewImage)",
        "Batch of alpha masks for each image",
        "Body2COLMAP dataset metadata (connect to ExportCOLMAP or SaveDataset)",
        "Image warp data for Generate FirstLast (only when override_cam_from_mesh is enabled)"
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

                # Face landmark rendering (requires skeleton render modes)
                "face_landmarks": ("B2C_FACE_LANDMARKS", {
                    "tooltip": (
                        "Optional face landmarks from Detect Face Landmarks node. "
                        "When connected, face keypoints are rendered on skeleton modes."
                    )
                }),
                "face_mode": (["full", "points", "none"], {
                    "default": "full",
                    "tooltip": (
                        "Face rendering mode: "
                        "full = points + connectivity lines, "
                        "points = points only, "
                        "none = disabled"
                    )
                }),
                "face_max_angle": ("FLOAT", {
                    "default": 90.0,
                    "min": 1.0,
                    "max": 90.0,
                    "step": 1.0,
                    "tooltip": (
                        "Max angle (degrees) between face normal and camera to render "
                        "face landmarks. 90 = full hemisphere, 45 = near-frontal only."
                    )
                }),

                # Point cloud sampling for COLMAP export
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh for COLMAP initialization"
                }),

                # Original camera override
                "override_cam_from_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use the original camera position from the mesh data instead of "
                        "path elevation/azimuth/radius. Works with Circular and Helical "
                        "Path (circular anchors frame 0, helical solves for the anchor "
                        "frame). Enables B2C_IMAGE_WARP output for the Generate FirstLast "
                        "node."
                    )
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
               face_landmarks=None, face_mode="full",
               face_max_angle=90.0,
               pointcloud_samples=10000,
               override_cam_from_mesh=False):
        """
        Render all camera positions and return batch of images + masks.

        Returns:
            images: Tensor of shape [N, H, W, 3] in [0,1] range (ComfyUI IMAGE format)
            masks: Tensor of shape [N, H, W] in [0,1] range (alpha channel)
            b2c_data: B2C_COLMAP_METADATA with cameras, point cloud, and image names
            image_warp: B2C_IMAGE_WARP warp data (only when override_cam_from_mesh=True)
        """
        # Get framing preset from path config
        framing = path_config.get("framing", "full")
        pattern = path_config["pattern"]

        # Validate override_cam_from_mesh constraints.  Circular anchors frame
        # 0 (constant elevation, so any frame works); helical sweeps elevation
        # and solves for the one frame that can land on the original camera.
        # Sinusoidal is deliberately unanchored in body2colmap.
        if override_cam_from_mesh and pattern not in ("circular", "helical"):
            raise ValueError(
                f"override_cam_from_mesh only works with Circular Path and "
                f"Helical Path, got '{pattern}' path."
            )

        # Convert SAM3D output to Scene
        # Always load skeleton if available to compute all framing bounds for metadata
        include_skeleton = True
        logger.info("[Body2COLMAP] Converting SAM3D output to scene...")
        t0 = time.time()
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=include_skeleton)
        logger.info(f"[Body2COLMAP] Scene conversion complete ({time.time() - t0:.2f}s)")

        # Extract original focal length from mesh data (needed for override mode)
        original_focal_length = float(mesh_data["focal_length"]) if override_cam_from_mesh else None

        if override_cam_from_mesh:
            # In override mode, skip auto-orient to preserve the original
            # camera-mesh relationship. The orbit parameters are derived
            # from the mesh's position relative to the origin (original camera).
            logger.info(
                "[Body2COLMAP] override_cam_from_mesh=True: skipping auto-orient, "
                "deriving orbit from original camera position"
            )
        else:
            # Auto-orient: rotate body to face camera at frame 0, then apply user offset
            initial_rotation = path_config.get("initial_rotation", 0.0)
            facing = scene.compute_torso_facing_direction()
            if facing is not None:
                current_angle = float(np.arctan2(facing[0], facing[2]))
                target_angle = float(np.arctan2(0.0, -1.0))  # face -Z (toward camera)
                correction_deg = float(np.degrees(target_angle - current_angle))
            else:
                correction_deg = 0.0
            total_rotation = correction_deg + initial_rotation
            scene.rotate_around_y(total_rotation)
            if facing is not None:
                logger.info(
                    f"[Body2COLMAP] Auto-orient: correction={correction_deg:.1f}° + "
                    f"offset={initial_rotation:.1f}° = {total_rotation:.1f}°"
                )
            elif initial_rotation != 0.0:
                logger.info(
                    f"[Body2COLMAP] No skeleton for auto-orient, "
                    f"applying raw rotation={initial_rotation:.1f}°"
                )

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

        # --- Camera path generation ---
        image_warp = None
        anchor_frame_index = None

        if override_cam_from_mesh:
            # Original-camera mode: derive orbit parameters from mesh geometry
            framing_info = compute_original_view_framing(
                vertices=scene.vertices,
                render_size=(width, height),
                original_focal_length=original_focal_length,
                fill_ratio=fill_ratio,
            )
            framed_fl = framing_info['framed_focal_length']

            camera_template = Camera(
                focal_length=(framed_fl, framed_fl),
                image_size=(width, height)
            )

            if pattern == "circular":
                orbit_params = compute_original_camera_orbit_params(orbit_center)
                derived_radius = float(orbit_params['radius'])
                # The orbit azimuth that puts the camera back at the original
                # camera position.  For circular that is also the start azimuth.
                anchor_azimuth = float(orbit_params['start_azimuth_deg'])
                derived_elevation = orbit_params['elevation_deg']
                anchor_frame_index = 0

                logger.info(
                    f"[Body2COLMAP] Original camera orbit (circular): "
                    f"radius={derived_radius:.3f}, azimuth={anchor_azimuth:.1f}°, "
                    f"elevation={derived_elevation:.1f}°, framed_fl={framed_fl:.1f}px"
                )

                path_gen = OrbitPath(target=orbit_center, radius=derived_radius)
                cameras = path_gen.circular(
                    n_frames=params["n_frames"],
                    elevation_deg=derived_elevation,
                    start_azimuth_deg=anchor_azimuth,
                    overlap=params.get("overlap", 1),
                    camera_template=camera_template,
                )
            else:
                # Helical: elevation sweeps, so the original camera can only be
                # reached at the frame whose elevation matches it.  Solve for
                # that frame, the start azimuth that lands on it, and the small
                # uniform elevation shift that makes it exact.
                #
                # Forward the path node's actual lead-in/lead-out (it defaults
                # to 30/90, not the solver's 45/45) or the solved index is wrong.
                helix_params = dict(
                    n_frames=params["n_frames"],
                    n_loops=params["n_loops"],
                    amplitude_deg=params["amplitude_deg"],
                    lead_in_deg=params.get("lead_in_deg", 45.0),
                    lead_out_deg=params.get("lead_out_deg", 45.0),
                )
                anchor_info = compute_helical_anchor_params(
                    target=orbit_center, **helix_params
                )
                derived_radius = float(anchor_info['radius'])
                anchor_azimuth = float(anchor_info['anchor_azimuth_deg'])
                anchor_frame_index = int(anchor_info['anchor_frame_index'])

                logger.info(
                    f"[Body2COLMAP] Original camera orbit (helical): "
                    f"radius={derived_radius:.3f}, anchor_frame={anchor_frame_index}, "
                    f"anchor_azimuth={anchor_azimuth:.1f}°, "
                    f"anchor_elevation={anchor_info['anchor_elevation_deg']:.2f}°, "
                    f"elevation_offset={anchor_info['elevation_offset_deg']:+.3f}°, "
                    f"framed_fl={framed_fl:.1f}px"
                )

                path_gen = OrbitPath(target=orbit_center, radius=derived_radius)
                cameras = path_gen.helical(
                    start_azimuth_deg=anchor_info['start_azimuth_deg'],
                    elevation_offset_deg=anchor_info['elevation_offset_deg'],
                    camera_template=camera_template,
                    **helix_params
                )

            # Build image warp data for Generate FirstLast.  The anchor frame
            # is the one sitting on the original camera — 0 for circular,
            # solved for on helical.
            image_warp = {
                "camera": cameras[anchor_frame_index],
                "original_focal_length": original_focal_length,
                "render_size": (width, height),
            }

            # Use framed focal length for downstream
            focal_length = framed_fl
        else:
            # Standard mode: use path config parameters
            # Determine focal length in pixels
            if focal_length_mm <= 0:
                focal_length = compute_default_focal_length(width)
            else:
                focal_length = focal_length_mm_to_pixels(focal_length_mm, width)

            # Auto-compute radius if not specified in path config
            if params.get("radius") is None:
                params["radius"] = compute_auto_orbit_radius(
                    bounds=current_bounds,
                    render_size=(width, height),
                    focal_length=focal_length,
                    fill_ratio=fill_ratio
                )

            camera_template = Camera(
                focal_length=(focal_length, focal_length),
                image_size=(width, height)
            )

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

        logger.info(f"[Body2COLMAP] Camera path created: {len(cameras)} cameras")

        # Prepare render colors
        mesh_color = (mesh_color_r, mesh_color_g, mesh_color_b)
        bg_color = (bg_color_r, bg_color_g, bg_color_b)

        # Attach bg_color to image_warp so GenerateFirstLast can use it
        # for out-of-bounds fill.
        if image_warp is not None:
            image_warp["bg_color"] = bg_color

        # Map "grayscale" to None (no colormap = grayscale depth)
        depth_cmap = None if depth_colormap == "grayscale" else depth_colormap

        # Convert face landmarks if provided
        openpose_face_70 = None
        effective_face_mode = None
        if face_landmarks is not None and face_mode != "none":
            source = face_landmarks["source"]
            if source == "mediapipe":
                logger.info("[Body2COLMAP] Converting MediaPipe face landmarks to OpenPose Face 70...")
                openpose_face_70 = FaceLandmarkIngest.from_mediapipe(
                    face_landmarks["landmarks"],
                    image_size=face_landmarks["image_size"],
                )
                effective_face_mode = face_mode
                logger.info(
                    f"[Body2COLMAP] Face landmarks converted: "
                    f"{openpose_face_70.shape}, face_mode={face_mode}"
                )
            else:
                raise ValueError(
                    f"Unsupported face landmark source: '{source}'. "
                    f"Supported: 'mediapipe'"
                )

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
                    bg_color=bg_color,
                    face_mode=effective_face_mode,
                    face_landmarks=openpose_face_70,
                    face_max_angle=face_max_angle,
                )
            elif render_mode == "mesh+skeleton":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_composite (mesh+skeleton)...")
                composite_modes = {
                    "mesh": {"color": mesh_color, "bg_color": bg_color},
                    "skeleton": {
                        "target_format": skeleton_format,
                        "joint_radius": joint_radius,
                        "bone_radius": bone_radius
                    }
                }
                if effective_face_mode is not None:
                    composite_modes["face"] = {
                        "face_mode": effective_face_mode,
                        "face_landmarks": openpose_face_70,
                        "face_max_angle": face_max_angle,
                    }
                img = renderer.render_composite(
                    camera=camera,
                    modes=composite_modes,
                )
            elif render_mode == "depth+skeleton":
                if i == 0:
                    logger.info("[Body2COLMAP] Calling render_composite (depth+skeleton)...")
                composite_modes = {
                    "depth": {"colormap": depth_cmap},
                    "skeleton": {
                        "target_format": skeleton_format,
                        "joint_radius": joint_radius,
                        "bone_radius": bone_radius
                    }
                }
                if effective_face_mode is not None:
                    composite_modes["face"] = {
                        "face_mode": effective_face_mode,
                        "face_landmarks": openpose_face_70,
                        "face_max_angle": face_max_angle,
                    }
                img = renderer.render_composite(
                    camera=camera,
                    modes=composite_modes,
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

        # Determine forward azimuth: the orbit azimuth that corresponds to
        # looking at the front of the skeleton.
        if override_cam_from_mesh:
            # The original camera was at the origin looking at the subject.
            # anchor_azimuth is the orbit azimuth that places the camera
            # back at the origin, so it equals "front".
            forward_azimuth_deg = float(anchor_azimuth)
        else:
            # After auto-orient the skeleton faces -Z, and OrbitPath azimuth
            # 0° = -Z direction, so forward is always 0° regardless of
            # start_azimuth_deg.
            forward_azimuth_deg = 0.0

        # Effective focal length to publish.  In override mode the widget value
        # is bypassed entirely (the render used the auto-framed focal length),
        # so publishing the raw widget would make downstream renders silently
        # reframe.  Convert the pixels we actually used back to mm.
        if override_cam_from_mesh:
            effective_focal_length_mm = focal_length_pixels_to_mm(focal_length, width)
        else:
            effective_focal_length_mm = focal_length_mm

        # Package metadata for serialization (no scene object - not serializable)
        b2c_data = {
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": (points, colors),
            "resolution": (width, height),
            "focal_length_mm": effective_focal_length_mm,  # 0 = auto, >0 = explicit 35mm equivalent
            "framing_bounds": all_framing_bounds,  # Dict of all computed framing bounds
            "initial_rotation": path_config.get("initial_rotation", 0.0),  # For splat renderer to reuse
            "orbit_target": orbit_center,  # np.ndarray(3,) — orbit center point
            "forward_azimuth_deg": forward_azimuth_deg,  # Orbit azimuth that = skeleton front
        }

        if override_cam_from_mesh:
            # Which rendered frame sits on the original camera, so downstream
            # stages know where the reference photo's viewpoint lives.
            b2c_data["anchor_frame_index"] = anchor_frame_index
            # The SAM-3D focal length in pixels.  Its presence also marks this
            # dataset as rendered *without* auto-orient, i.e. the original
            # camera really is at the world origin — the invariant the splat
            # renderer's own override mode depends on.
            b2c_data["original_focal_length"] = original_focal_length

        return (images_tensor, masks_tensor, b2c_data, image_warp)
