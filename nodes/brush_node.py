"""Brush CLI node - trains 3D Gaussian Splats using the brush application."""

import subprocess
import tempfile
import shutil
import logging
import time
import threading
from pathlib import Path
import numpy as np
import cv2
import gc

import torch
import comfy.model_management as model_management
import comfy.utils

from body2colmap.exporter import ColmapExporter
from body2colmap.splat_scene import SplatScene
from ..core.comfy_utils import comfy_to_cv2

logger = logging.getLogger(__name__)


class Body2COLMAP_RunBrush:
    """Train a 3D Gaussian Splat using the brush CLI tool."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "run_brush"
    RETURN_TYPES = ("SPLAT_SCENE", "B2C_COLMAP_METADATA")
    RETURN_NAMES = ("splat_scene", "b2c_data")
    OUTPUT_TOOLTIPS = (
        "Trained Gaussian splat scene",
        "Updated B2C metadata with splat reference (use Save Dataset to persist)"
    )

    # Tell ComfyUI to collect all batch outputs into lists
    INPUT_IS_LIST = {
        "images": True,
        "masks": True,
    }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute training (never use cached results)
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "b2c_data": ("B2C_COLMAP_METADATA", {
                    "tooltip": "Dataset metadata from render or load nodes"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Rendered images for training"
                }),
                "brush_path": ("STRING", {
                    "default": "brush",
                    "tooltip": "Path to the brush executable (or 'brush' if in PATH)"
                }),
                "total_steps": ("INT", {
                    "default": 30000,
                    "min": 100,
                    "max": 100000,
                    "step": 100,
                    "tooltip": "Total training iterations"
                }),
                "sh_degree": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 4,
                    "tooltip": "Spherical harmonics degree"
                }),
                "merge_batches": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Merge batched inputs into single dataset (enable when loading with batch_size > 0)"
                }),
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": "Optional masks for alpha channel"
                }),
                "unload_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload ComfyUI models before training to free VRAM"
                }),
                "with_viewer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Spawn viewer during training"
                }),
                "max_resolution": ("INT", {
                    "default": 1920,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Maximum image resolution for training"
                }),
                "max_splats": ("INT", {
                    "default": 10000000,
                    "min": 100000,
                    "max": 50000000,
                    "step": 100000,
                    "tooltip": "Maximum number of Gaussian splats"
                }),
                "refine_every": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Refinement frequency (steps between densification)"
                }),
                "alpha_mode": (["masked", "transparent"], {
                    "default": "transparent",
                    "tooltip": "How to interpret alpha channel in images"
                }),
            }
        }

    def run_brush(
        self,
        b2c_data,
        images,
        brush_path,
        total_steps,
        sh_degree,
        merge_batches=False,
        masks=None,
        unload_models=True,
        with_viewer=False,
        max_resolution=1920,
        max_splats=10000000,
        refine_every=200,
        alpha_mode="transparent",
    ):
        """
        Execute brush training on the provided dataset.

        Args:
            b2c_data: B2C_COLMAP_METADATA with cameras, image_names, points_3d
            images: ComfyUI IMAGE tensor or List[IMAGE] when batched
            brush_path: Path to brush executable
            total_steps: Number of training iterations
            sh_degree: Spherical harmonics degree
            merge_batches: If True, merge batched inputs into single dataset
            masks: Optional MASK tensor or List[MASK] when batched
            unload_models: Unload ComfyUI models before training
            with_viewer: Spawn viewer during training
            max_resolution: Maximum image resolution
            max_splats: Maximum number of splats
            refine_every: Refinement frequency
            alpha_mode: How to interpret alpha channel

        Returns:
            Tuple of (splat_scene, updated_b2c_data)
        """
        # Unwrap scalar parameters if they come as lists (happens when INPUT_IS_LIST is set)
        # When INPUT_IS_LIST is present, ComfyUI passes all inputs as lists in batched contexts
        if isinstance(b2c_data, list):
            b2c_data = b2c_data[0]
        if isinstance(brush_path, list):
            brush_path = brush_path[0]
        if isinstance(total_steps, list):
            total_steps = total_steps[0]
        if isinstance(sh_degree, list):
            sh_degree = sh_degree[0]
        if isinstance(merge_batches, list):
            merge_batches = merge_batches[0]
        if isinstance(unload_models, list):
            unload_models = unload_models[0]
        if isinstance(with_viewer, list):
            with_viewer = with_viewer[0]
        if isinstance(max_resolution, list):
            max_resolution = max_resolution[0]
        if isinstance(max_splats, list):
            max_splats = max_splats[0]
        if isinstance(refine_every, list):
            refine_every = refine_every[0]
        if isinstance(alpha_mode, list):
            alpha_mode = alpha_mode[0]

        # Handle batch merging
        if merge_batches:
            # Concatenate all batches into single tensor
            if isinstance(images, list) and len(images) > 1:
                images = torch.cat(images, dim=0)
                logger.info(f"[Body2COLMAP] Merged {len(images)} image batches")
            elif isinstance(images, list):
                images = images[0]  # Single batch

            if masks is not None:
                if isinstance(masks, list) and len(masks) > 1:
                    masks = torch.cat(masks, dim=0)
                    logger.info(f"[Body2COLMAP] Merged {len(masks)} mask batches")
                elif isinstance(masks, list):
                    masks = masks[0]  # Single batch
        else:
            # Extract single batch (backward compatible)
            if isinstance(images, list):
                if len(images) > 1:
                    raise ValueError(
                        f"Received {len(images)} batches but merge_batches=False. "
                        "Enable merge_batches or disable batching in Load Dataset (set batch_size=0)"
                    )
                images = images[0]

            if masks is not None and isinstance(masks, list):
                if len(masks) > 1:
                    raise ValueError(
                        f"Received {len(masks)} mask batches but merge_batches=False. "
                        "Enable merge_batches or disable batching in Load Dataset (set batch_size=0)"
                    )
                masks = masks[0]

        # 1. Create temporary directory for brush output (persists after function returns)
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        temp_output = Path("temp") / "brush" / f"training_{timestamp}"
        temp_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Body2COLMAP] Brush temporary output: {temp_output}")

        # 2. Create temporary COLMAP directory
        with tempfile.TemporaryDirectory(prefix="b2c_colmap_") as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"[Body2COLMAP] Temporary COLMAP directory: {temp_path}")

            # 3. Export COLMAP format
            logger.info("[Body2COLMAP] Exporting to COLMAP format...")
            exporter = ColmapExporter(
                cameras=b2c_data["cameras"],
                image_names=b2c_data["image_names"],
                points_3d=b2c_data["points_3d"]
            )
            exporter.export(output_dir=temp_path)

            # 4. Export images with optional alpha channel
            images_dir = temp_path / "images"
            images_dir.mkdir(exist_ok=True)

            logger.info(f"[Body2COLMAP] Exporting {len(images)} images...")

            # Convert ComfyUI images to cv2 format
            cv2_images = comfy_to_cv2(images)

            # Save images (with alpha channel if masks provided)
            if masks is not None:
                # Convert masks from ComfyUI format [B, H, W] float [0,1] to [B, H, W] uint8 [0,255]
                # Note: ComfyUI MASK is inverted (1.0 = background), so we invert back for alpha channel
                masks_np = masks.cpu().numpy()
                alpha_channel = ((1.0 - masks_np) * 255).astype(np.uint8)

                # Save RGBA
                for i, (img, filename) in enumerate(zip(cv2_images, b2c_data["image_names"])):
                    alpha = alpha_channel[i]  # [H, W]

                    # Check if image already has alpha channel (4 channels)
                    if img.shape[-1] == 4:
                        # Replace existing alpha channel with our mask
                        rgba = img.copy()
                        rgba[..., 3] = alpha
                    elif img.shape[-1] == 3:
                        # Add alpha channel to BGR image
                        rgba = np.dstack([img, alpha])  # [H, W, 4] - BGRA
                    else:
                        raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")

                    img_path = images_dir / filename
                    cv2.imwrite(str(img_path), rgba)
            else:
                # Save images without modifying alpha channel (RGB or RGBA as-is)
                for img, filename in zip(cv2_images, b2c_data["image_names"]):
                    img_path = images_dir / filename
                    # Image can be BGR (3 channels) or BGRA (4 channels)
                    cv2.imwrite(str(img_path), img)

            logger.info(f"[Body2COLMAP] Exported {len(cv2_images)} images to {images_dir}")

            # 5. Optionally unload ComfyUI models
            if unload_models:
                self._unload_comfy_models()

            # 6. Build brush command
            ply_output_name = "export.ply"
            cmd = [
                brush_path,
                str(temp_path),
                "--total-steps", str(total_steps),
                "--sh-degree", str(sh_degree),
                "--export-path", str(temp_output.absolute()),
                "--export-name", ply_output_name,
                "--export-every", str(total_steps),  # Only export at end
                "--max-resolution", str(max_resolution),
                "--max-splats", str(max_splats),
                "--refine-every", str(refine_every),
            ]

            if with_viewer:
                cmd.append("--with-viewer")

            if masks is not None:
                cmd.extend(["--alpha-mode", alpha_mode])

            # 7. Execute brush
            logger.info(f"[Body2COLMAP] Running brush: {' '.join(cmd)}")
            print(f"[Body2COLMAP] Starting brush training ({total_steps} steps)...")

            # Create progress bar (needed for interrupt checking)
            pbar = comfy.utils.ProgressBar(total_steps)

            process = None
            try:
                # Use Popen to stream output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffered
                    cwd=str(Path.cwd())
                )

                output_lines = []

                # Thread to read output without blocking
                def read_output():
                    try:
                        for line in process.stdout:
                            output_lines.append(line)
                            logger.debug(f"[Brush] {line.rstrip()}")
                    except:
                        pass

                output_thread = threading.Thread(target=read_output, daemon=True)
                output_thread.start()

                # Poll the process and check for interrupts periodically
                while True:
                    # Check if process has finished
                    return_code = process.poll()
                    if return_code is not None:
                        break

                    # Allow interrupt checking by updating progress bar
                    pbar.update(0)

                    # Sleep briefly before next check
                    time.sleep(0.5)

                # Wait for output thread to finish
                output_thread.join(timeout=1.0)

                if return_code != 0:
                    logger.error("[Body2COLMAP] Brush failed")
                    logger.error(f"[Body2COLMAP] Output:\n{''.join(output_lines)}")
                    raise RuntimeError(
                        f"Brush training failed with exit code {return_code}.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"Check logs for details."
                    )

                logger.info("[Body2COLMAP] Brush training completed successfully")
                print("[Body2COLMAP] Brush training completed successfully")

                # Log full output at debug level
                logger.debug(f"[Body2COLMAP] Brush output:\n{''.join(output_lines)}")

            except (KeyboardInterrupt, comfy.model_management.InterruptProcessingException):
                # User cancelled - kill the subprocess
                logger.info("[Body2COLMAP] Training cancelled by user, terminating brush process...")
                print("[Body2COLMAP] Training cancelled, terminating brush process...")
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("[Body2COLMAP] Brush did not terminate, killing forcefully...")
                        process.kill()
                        process.wait()
                raise
            except Exception as e:
                # Any other exception - make sure to clean up subprocess
                if process is not None and process.poll() is None:
                    logger.warning("[Body2COLMAP] Exception occurred, terminating brush process...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                if isinstance(e, FileNotFoundError):
                    raise RuntimeError(
                        f"Brush executable not found at: {brush_path}\n"
                        f"Please ensure brush is installed and the path is correct."
                    )

                logger.error(f"[Body2COLMAP] Unexpected error running brush: {e}")
                raise

        # 8. Load trained splat
        ply_path = temp_output / ply_output_name
        if not ply_path.exists():
            raise RuntimeError(
                f"Expected output PLY file not found: {ply_path}\n"
                f"Brush may not have exported successfully."
            )

        logger.info(f"[Body2COLMAP] Loading trained splat from {ply_path}")
        splat_scene = SplatScene.from_ply(str(ply_path))
        logger.info(f"[Body2COLMAP] Loaded splat with {len(splat_scene)} Gaussians")
        print(f"[Body2COLMAP] Trained splat: {len(splat_scene)} Gaussians, SH degree {splat_scene.sh_degree}")
        print(f"[Body2COLMAP] Temporary output: {temp_output}")
        print(f"[Body2COLMAP] Use Save Dataset to persist the trained splat")

        # 9. Update b2c_data with splat metadata
        updated_b2c_data = b2c_data.copy()
        updated_b2c_data["splat_path"] = str(ply_path.absolute())

        return (splat_scene, updated_b2c_data)

    def _unload_comfy_models(self):
        """Unload all ComfyUI models to free VRAM for brush training."""
        logger.info("[Body2COLMAP] Unloading all ComfyUI models...")
        print("[Body2COLMAP] Unloading models to free VRAM...")

        # Use ComfyUI's proper model management API
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)

        # Additional cleanup for thorough VRAM clearing
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[Body2COLMAP] Successfully unloaded models and cleared VRAM")
            print("[Body2COLMAP] Models unloaded, VRAM cleared")
        except Exception as e:
            logger.warning(f"[Body2COLMAP] Unable to fully clear cache: {e}")
