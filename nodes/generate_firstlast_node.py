"""Generate FirstLast node - warps reference image to match rendered skeleton."""

import logging

import cv2
import numpy as np
import torch
from body2colmap.utils import compute_warp_to_camera

logger = logging.getLogger(__name__)


class Body2COLMAP_GenerateFirstLast:
    """Warp a reference image to align with the skeleton rendered at frame 0.

    Takes B2C_IMAGE_WARP (from Render node with override_cam_from_mesh) and
    the original reference photo, and produces a warped version that matches
    the rendered skeleton's viewpoint.  Useful as diffusion conditioning for
    the first and last frames of a circular orbit.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_image",)
    OUTPUT_TOOLTIPS = (
        "Reference image warped to match the rendered skeleton at frame 0",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_warp": ("B2C_IMAGE_WARP",),
                "image": ("IMAGE",),
            },
        }

    def generate(self, image_warp, image):
        """Warp the reference image using the stored camera and focal length.

        The warp accounts for the focal-length zoom (auto-framing) and the
        slight rotation correction from look_at, transforming the original
        photo so it aligns pixel-for-pixel with the rendered skeleton at
        frame 0 of the orbit.

        Args:
            image_warp: B2C_IMAGE_WARP dict with camera, original_focal_length,
                        and render_size.
            image: ComfyUI IMAGE tensor [B, H, W, 3] float32 in [0, 1].

        Returns:
            Warped image as ComfyUI IMAGE tensor [1, H_out, W_out, 3].
        """
        camera = image_warp["camera"]
        original_focal_length = image_warp["original_focal_length"]
        render_w, render_h = image_warp["render_size"]

        # Convert first image from ComfyUI format to uint8 RGB
        img_np = image[0].cpu().numpy()  # [H, W, 3] float32 [0,1]
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        h_img, w_img = img_uint8.shape[:2]

        # Scale the focal length to match the actual reference image
        # dimensions.  The SAM3D focal_length corresponds to the render
        # resolution; if the reference image is a different size we need
        # to adjust so the intrinsics matrix K_orig is consistent.
        scale = w_img / render_w
        scaled_focal_length = original_focal_length * scale

        logger.info(
            f"[Body2COLMAP] GenerateFirstLast: ref={w_img}x{h_img}, "
            f"render={render_w}x{render_h}, fl_scale={scale:.3f}"
        )

        # Check whether the camera rotation is identity (pure zoom+shift)
        # or requires a full perspective warp.
        is_identity_rotation = np.allclose(
            camera.rotation, np.eye(3), atol=1e-5
        )

        border_color = (255, 255, 255)  # white fill for out-of-bounds pixels

        if is_identity_rotation:
            # Pure affine: scale + translate (fast path)
            s = float(camera.fx / scaled_focal_length)
            tx = camera.cx - s * w_img / 2.0
            ty = camera.cy - s * h_img / 2.0
            M = np.array([[s, 0.0, tx],
                          [0.0, s, ty]], dtype=np.float64)
            warped = cv2.warpAffine(
                img_uint8, M, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )
        else:
            # Full homography: accounts for rotation + intrinsic change
            H = compute_warp_to_camera(
                original_focal_length=scaled_focal_length,
                original_image_size=(w_img, h_img),
                target_camera=camera,
            )
            warped = cv2.warpPerspective(
                img_uint8, H, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )

        # Convert back to ComfyUI IMAGE format: [1, H, W, 3] float32 [0,1]
        warped_float = warped.astype(np.float32) / 255.0
        warped_tensor = torch.from_numpy(warped_float).unsqueeze(0)

        return (warped_tensor,)
