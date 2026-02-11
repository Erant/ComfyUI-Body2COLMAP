"""Camera parameter utilities for focal length conversions."""

# 35mm full-frame sensor width in mm (standard reference for focal length)
FULL_FRAME_SENSOR_WIDTH_MM = 36.0


def focal_length_mm_to_pixels(focal_length_mm: float, image_width: int) -> float:
    """
    Convert focal length from millimeters (35mm full-frame equivalent) to pixels.

    Uses the standard 35mm full-frame sensor width (36mm) as reference.
    This allows photographers to use familiar focal lengths like 35mm, 50mm, 85mm.

    Args:
        focal_length_mm: Focal length in millimeters (35mm equivalent)
        image_width: Image width in pixels

    Returns:
        Focal length in pixels for the camera intrinsic matrix
    """
    return (focal_length_mm / FULL_FRAME_SENSOR_WIDTH_MM) * image_width


def focal_length_pixels_to_mm(focal_length_px: float, image_width: int) -> float:
    """
    Convert focal length from pixels to millimeters (35mm full-frame equivalent).

    Inverse of :func:`focal_length_mm_to_pixels`.

    Args:
        focal_length_px: Focal length in pixels
        image_width: Image width in pixels

    Returns:
        Focal length in millimeters (35mm full-frame equivalent)
    """
    return (focal_length_px / image_width) * FULL_FRAME_SENSOR_WIDTH_MM
