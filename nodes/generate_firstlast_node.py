"""Generate FirstLast node - warps reference image to match rendered skeleton."""

import logging

import cv2
import numpy as np
import torch
from body2colmap.utils import compute_warp_to_camera

logger = logging.getLogger(__name__)


class Body2COLMAP_GenerateFirstLast:
    """Warp a reference image to align with the skeleton at the anchor frame.

    Takes B2C_IMAGE_WARP (from Render node with override_cam_from_mesh) and
    the original reference photo, and produces a warped version that matches
    the rendered skeleton's viewpoint.  Useful as diffusion conditioning for
    the orbit frame(s) sitting at the original camera — frame 0 and the last
    frame of a circular orbit, or the solved-for frame of a helical one.
    Feed the result to Inject Anchor to write it into the batch.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_image",)
    OUTPUT_TOOLTIPS = (
        "Reference image warped to match the rendered skeleton at the anchor frame",
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
        photo so it aligns pixel-for-pixel with the rendered skeleton at the
        orbit's anchor frame.

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

        # The SAM3D focal_length is in pixels for the original photo
        # resolution.  compute_warp_to_camera builds K_orig from this FL
        # and original_image_size, so the resolution difference between
        # the reference image and the render is handled automatically by
        # the homography K_target @ R @ inv(K_orig).  No manual scaling.

        logger.info(
            f"[Body2COLMAP] GenerateFirstLast: ref={w_img}x{h_img}, "
            f"render={render_w}x{render_h}, fl={original_focal_length:.1f}"
        )

        # Check whether the camera rotation is identity (pure zoom+shift)
        # or requires a full perspective warp.
        is_identity_rotation = np.allclose(
            camera.rotation, np.eye(3), atol=1e-5
        )

        # Background color: use the mesh render bg_color when available,
        # fall back to white.
        bg_rgb = image_warp.get("bg_color", (1.0, 1.0, 1.0))
        border_color = (
            int(round(bg_rgb[0] * 255)),
            int(round(bg_rgb[1] * 255)),
            int(round(bg_rgb[2] * 255)),
        )

        if is_identity_rotation:
            # Pure affine: scale + translate (fast path)
            s = float(camera.fx / original_focal_length)
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
                original_focal_length=original_focal_length,
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
