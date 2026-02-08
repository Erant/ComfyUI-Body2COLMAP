"""Face landmark detection node using MediaPipe FaceLandmarker.

Detects facial landmarks from an input image and outputs them in raw
MediaPipe format for downstream consumption by body2colmap's
FaceLandmarkIngest.

Detection pipeline (lifted from body2colmap tools/extract_face_landmarks.py):
1. Try FaceLandmarker on full image (fast path for selfies/headshots)
2. If no face found, use FaceDetector to locate bounding boxes, crop with
   configurable padding, and re-run FaceLandmarker on the crop
3. If multiple faces, pick the most frontal via cross-product frontality score
4. Landmarks are mapped back to full-image normalized coordinates
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np

from ..core.comfy_utils import comfy_to_rgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model management (matches body2colmap tools/extract_face_landmarks.py)
# ---------------------------------------------------------------------------
LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "body2colmap"
LANDMARKER_MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"
DETECTOR_MODEL_PATH = MODEL_CACHE_DIR / "blaze_face_short_range.tflite"


def _ensure_model(url: str, path: Path) -> Path:
    """Download a model file if not cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Body2COLMAP] Downloading {path.name}...")
    urllib.request.urlretrieve(url, str(path))
    logger.info(f"[Body2COLMAP] Downloaded {path.name}")
    return path


# ---------------------------------------------------------------------------
# MediaPipe landmark indices for frontality scoring
# ---------------------------------------------------------------------------
_MP_RIGHT_EYE_OUTER = 33
_MP_LEFT_EYE_OUTER = 263
_MP_NOSE_BRIDGE = 168
_MP_CHIN = 152


def _frontality_score(face_landmarks):
    """Score how frontal a face is (0 = profile, 1 = frontal)."""
    def _xyz(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    eye_vec = _xyz(_MP_LEFT_EYE_OUTER) - _xyz(_MP_RIGHT_EYE_OUTER)
    vert_vec = _xyz(_MP_NOSE_BRIDGE) - _xyz(_MP_CHIN)
    normal = np.cross(eye_vec, vert_vec)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return 0.0
    return abs(normal[2]) / norm


def _pick_best_face(face_landmarks_list):
    """Pick the most frontal face from a list of detected faces."""
    if len(face_landmarks_list) == 1:
        return face_landmarks_list[0], 0

    best_score = -1.0
    best_idx = 0
    for i, face in enumerate(face_landmarks_list):
        score = _frontality_score(face)
        if score > best_score:
            best_score = score
            best_idx = i
    return face_landmarks_list[best_idx], best_idx


def _find_all_face_bboxes(rgb, detector, mp_module):
    """Find all face bounding boxes using MediaPipe FaceDetector."""
    image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    bboxes = []
    for det in result.detections:
        bbox = det.bounding_box
        bboxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
    return bboxes


def _crop_to_face(rgb, bbox, padding=0.5):
    """Crop image to face bounding box with padding."""
    height, width = rgb.shape[:2]
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2, :])
    return crop, x1, y1


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class Body2COLMAP_DetectFaceLandmarks:
    """Detect face landmarks from an image using MediaPipe FaceLandmarker.

    Outputs raw MediaPipe landmarks (478 or 468 points) as a
    B2C_FACE_LANDMARKS dict for downstream use by body2colmap's
    FaceLandmarkIngest.
    """

    CATEGORY = "Body2COLMAP"
    FUNCTION = "detect"
    RETURN_TYPES = ("B2C_FACE_LANDMARKS",)
    RETURN_NAMES = ("face_landmarks",)
    OUTPUT_TOOLTIPS = (
        "Face landmarks: raw MediaPipe points for body2colmap ingestion",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (first frame of batch is used)"
                }),
            },
            "optional": {
                "min_detection_confidence": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum confidence for face detection (0-1)"
                }),
                "crop_padding": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": (
                        "Padding around detected face bbox as a fraction of "
                        "face size, used in the fallback crop-and-retry stage"
                    )
                }),
            }
        }

    def detect(self, image, min_detection_confidence=0.3, crop_padding=0.5):
        """
        Detect face landmarks using MediaPipe FaceLandmarker.

        Two-stage pipeline:
        1. Run FaceLandmarker on full image
        2. Fallback: FaceDetector -> crop with padding -> re-run FaceLandmarker

        Returns:
            Tuple of (B2C_FACE_LANDMARKS,)
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for face landmark detection. "
                "Install with: pip install mediapipe"
            ) from e

        # Convert first image from batch to RGB uint8
        rgb_images = comfy_to_rgb(image)
        rgb = rgb_images[0]
        height, width = rgb.shape[:2]

        logger.info(
            f"[Body2COLMAP] Detecting face landmarks "
            f"(image: {width}x{height}, confidence: {min_detection_confidence})"
        )

        # Ensure models are available
        landmarker_path = str(
            _ensure_model(LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)
        )
        detector_path = str(
            _ensure_model(DETECTOR_MODEL_URL, DETECTOR_MODEL_PATH)
        )

        # Create FaceLandmarker
        lm_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=landmarker_path
            ),
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            num_faces=10,
        )
        landmarker = vision.FaceLandmarker.create_from_options(lm_options)

        try:
            landmarks_array = self._run_pipeline(
                rgb, width, height,
                landmarker, detector_path,
                min_detection_confidence, crop_padding,
                mp, vision, python,
            )
        finally:
            landmarker.close()

        face_landmarks_data = {
            "source": "mediapipe",
            "landmarks": landmarks_array,
            "image_size": (width, height),
        }

        logger.info(
            f"[Body2COLMAP] Face landmarks output: "
            f"{landmarks_array.shape[0]} points, "
            f"source=mediapipe"
        )
        return (face_landmarks_data,)

    def _run_pipeline(
        self, rgb, width, height,
        landmarker, detector_path,
        min_confidence, crop_padding,
        mp, vision, python,
    ):
        """Two-stage face landmark detection pipeline.

        Returns:
            np.ndarray of shape (N, 3) with normalized landmark coords.
        """
        # Stage 1: Try full image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            face, idx = _pick_best_face(result.face_landmarks)
            if len(result.face_landmarks) > 1:
                score = _frontality_score(face)
                logger.info(
                    f"[Body2COLMAP] Found {len(result.face_landmarks)} face(s), "
                    f"selected #{idx + 1} (frontality: {score:.2f})"
                )
            return self._face_to_array(face)

        # Stage 2: FaceDetector crop-and-retry
        logger.info(
            "[Body2COLMAP] No face found on full image, "
            "trying FaceDetector crop fallback..."
        )

        det_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=detector_path
            ),
            min_detection_confidence=min_confidence,
        )
        detector = vision.FaceDetector.create_from_options(det_options)

        try:
            bboxes = _find_all_face_bboxes(rgb, detector, mp)
        finally:
            detector.close()

        if not bboxes:
            raise RuntimeError(
                f"No face detected in image ({width}x{height}). "
                f"Ensure the image contains a visible face."
            )

        logger.info(
            f"[Body2COLMAP] FaceDetector found {len(bboxes)} face(s), "
            f"cropping with padding={crop_padding}"
        )

        # Extract landmarks from each crop
        candidates = []
        for bbox in bboxes:
            crop, x1, y1 = _crop_to_face(rgb, bbox, padding=crop_padding)
            crop_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=crop
            )
            crop_result = landmarker.detect(crop_image)
            if crop_result.face_landmarks:
                candidates.append(
                    (crop_result.face_landmarks[0], crop, x1, y1)
                )

        if not candidates:
            raise RuntimeError(
                f"FaceDetector located {len(bboxes)} face(s) but "
                f"FaceLandmarker could not extract landmarks from any crop."
            )

        # Pick the most frontal face
        faces_only = [c[0] for c in candidates]
        _, best_idx = _pick_best_face(faces_only)
        face, crop, x1, y1 = candidates[best_idx]
        crop_h, crop_w = crop.shape[:2]

        if len(candidates) > 1:
            score = _frontality_score(face)
            logger.info(
                f"[Body2COLMAP] Selected face #{best_idx + 1} of "
                f"{len(candidates)} (frontality: {score:.2f})"
            )

        logger.info(
            f"[Body2COLMAP] Face extracted from crop "
            f"({crop_w}x{crop_h} at offset {x1},{y1})"
        )

        # Map crop-normalized landmarks back to full-image normalized coords
        return self._face_to_array_from_crop(
            face, crop_w, crop_h, x1, y1, width, height
        )

    @staticmethod
    def _face_to_array(face_landmarks):
        """Convert MediaPipe face landmarks to (N, 3) numpy array."""
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks],
            dtype=np.float32,
        )

    @staticmethod
    def _face_to_array_from_crop(
        face_landmarks, crop_w, crop_h, x1, y1, full_w, full_h
    ):
        """Convert crop-space landmarks to full-image normalized coords."""
        return np.array(
            [
                [
                    (lm.x * crop_w + x1) / full_w,
                    (lm.y * crop_h + y1) / full_h,
                    lm.z,
                ]
                for lm in face_landmarks
            ],
            dtype=np.float32,
        )
