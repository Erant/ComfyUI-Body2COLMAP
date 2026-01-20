"""Export node for Body2COLMAP - exports COLMAP sparse reconstruction format."""

from pathlib import Path
import cv2
import numpy as np
from body2colmap.exporter import ColmapExporter
from ..core.comfy_utils import comfy_to_cv2


class Body2COLMAP_ExportCOLMAP:
    """Export COLMAP sparse reconstruction format for 3DGS training."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True  # Terminal node - produces file output
    OUTPUT_TOOLTIPS = ("Path to the output directory containing COLMAP files",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_data": ("B2C_RENDER_DATA",),
                "output_directory": ("STRING", {
                    "default": "output/colmap",
                    "tooltip": "Directory for COLMAP files and images (flat structure)"
                }),
                "image_name": ("STRING", {
                    "default": "frame",
                    "tooltip": "Base name for images (follows ComfyUI convention: <name>_%05d_.png)"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh surface"
                }),
            }
        }

    def export(self, render_data, output_directory, image_name,
               images=None, masks=None, pointcloud_samples=10000):
        """
        Export COLMAP format files.

        Creates (flat structure matching body2colmap):
            output_directory/
            ├── frame_00001_.png (RGBA if masks provided, RGB otherwise)
            ├── frame_00002_.png
            ├── ...
            ├── cameras.txt   (camera intrinsics)
            ├── images.txt    (camera extrinsics per image)
            └── points3D.txt  (initial point cloud)
        """
        # Extract data
        cameras = render_data["cameras"]
        scene = render_data["scene"]
        width, height = render_data["resolution"]
        focal_length = render_data["focal_length"]

        # Create output directory
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate image filenames using ComfyUI convention
        # Format: <image_name>_%05d_.png with 1-based indexing
        image_names = [f"{image_name}_{i+1:05d}_.png" for i in range(len(cameras))]

        # Save images if provided
        if images is not None:
            # Convert ComfyUI images to CV2 format
            cv2_images = comfy_to_cv2(images)

            # If masks provided, combine with images as alpha channel
            if masks is not None:
                # Convert masks from ComfyUI format [B, H, W] float [0,1] to [B, H, W] uint8 [0,255]
                # Note: ComfyUI MASK is inverted (1.0 = background), so we invert back for alpha channel
                masks_np = masks.cpu().numpy()
                alpha_channel = ((1.0 - masks_np) * 255).astype(np.uint8)  # Invert back for alpha

                # Combine RGB with alpha
                for i, (img, filename) in enumerate(zip(cv2_images, image_names)):
                    # img is BGR from comfy_to_cv2, shape [H, W, 3]
                    # Add alpha channel
                    alpha = alpha_channel[i]  # [H, W]
                    rgba = np.dstack([img, alpha])  # [H, W, 4] - BGRA

                    img_path = output_path / filename
                    cv2.imwrite(str(img_path), rgba)
            else:
                # Save RGB only
                for img, filename in zip(cv2_images, image_names):
                    img_path = output_path / filename
                    cv2.imwrite(str(img_path), img)

        # Create COLMAP exporter using classmethod
        exporter = ColmapExporter.from_scene_and_cameras(
            scene=scene,
            cameras=cameras,
            image_names=image_names,
            n_pointcloud_samples=pointcloud_samples
        )

        # Export COLMAP files to same directory as images
        exporter.export(output_dir=output_path)

        print(f"[Body2COLMAP] Exported COLMAP files to: {output_path}")
        print(f"[Body2COLMAP] - cameras.txt: {len(cameras)} cameras")
        print(f"[Body2COLMAP] - images.txt: {len(cameras)} images")
        print(f"[Body2COLMAP] - points3D.txt: {pointcloud_samples} points")

        if images is not None:
            format_str = "RGBA" if masks is not None else "RGB"
            print(f"[Body2COLMAP] - {len(cv2_images)} {format_str} image files")

        return (str(output_path.absolute()),)
