"""Landmark-based face cropping and jaw-contour blending for MuseTalk.

The musetalk_mlx package owns only the neural generation. Face detection, the
256x256 crop, and paste-back are caller-supplied. This mirrors the upstream
TMElyralab/MuseTalk preprocessing/blending convention, but torch-free:

- Detection uses MediaPipe FaceLandmarker (478 landmarks) instead of DWPose+S3FD.
- The crop bbox follows upstream `get_landmark_and_bbox`: x spans the face width,
  the box is vertically centred on a nose reference and reaches the chin, with a
  tunable `bbox_shift` (vertical nudge) and `extra_margin` (extra chin) — exactly
  the knobs upstream exposes.
- Paste-back blends only the lower face through a feathered convex-hull mask of the
  jaw/face-oval landmarks (the analogue of upstream's BiSeNet parse mask), so the
  jaw moves naturally and edges stay smooth.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

CROP_SIZE = 256

# MediaPipe FaceMesh face-oval (silhouette) landmark indices.
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
NOSE_REF = 6  # upper nose bridge (≈ between the eyes) — vertical crop centre
CHIN = 152  # lowest face-oval point

_MODEL_PATH = os.environ.get("MUSETALK_LANDMARKER", "models/face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_landmarker: FaceLandmarker | None = None


def _get_landmarker() -> FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(_MODEL_PATH):
            os.makedirs(os.path.dirname(_MODEL_PATH) or ".", exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        _landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH), num_faces=1
            )
        )
    return _landmarker


@dataclass
class FaceBox:
    """Crop region in the original frame (x1, y1, x2, y2); may be non-square."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


def detect_landmarks(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return (478, 2) pixel-space landmarks for the largest face, or None."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _get_landmarker().detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def box_from_landmarks(
    landmarks: np.ndarray,
    frame_shape: tuple[int, int],
    bbox_shift: int = 0,
    extra_margin: int = 10,
) -> FaceBox:
    """Upstream-style crop box: face width × (nose-centred, reaching chin).

    bbox_shift: pixels to move the nose reference down (+) / up (-), shrinking or
                growing the forehead side, controlling how much mouth is in frame.
    extra_margin: extra pixels below the chin (MuseTalk v1.5 convention).
    """
    h, w = frame_shape[:2]
    oval = landmarks[FACE_OVAL]
    x1 = int(np.min(oval[:, 0]))
    x2 = int(np.max(oval[:, 0]))
    chin_y = float(landmarks[CHIN][1])
    nose_y = float(landmarks[NOSE_REF][1]) + bbox_shift
    half = chin_y - nose_y
    y1 = int(nose_y - half)
    y2 = int(chin_y + extra_margin)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return FaceBox(x1, y1, x2, y2)


def crop_face(img_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
    """256x256 BGR crop (LANCZOS4, matching upstream resize)."""
    crop = img_bgr[box.y1 : box.y2, box.x1 : box.x2]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)


def jaw_mask(
    landmarks: np.ndarray,
    box: FaceBox,
    upper_ratio: float = 0.5,
    blur_frac: float = 0.08,
) -> np.ndarray:
    """Feathered alpha mask (box.h × box.w × 1) over the lower face.

    Convex hull of the face-oval landmarks (clipped to the box), with the top
    `upper_ratio` zeroed so only the talking region is blended, then Gaussian
    blurred — the torch-free analogue of upstream's parse-mask blend.
    """
    hw = (box.h, box.w)
    m = np.zeros(hw, dtype=np.float32)
    pts = (landmarks[FACE_OVAL] - [box.x1, box.y1]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(m, hull, 1.0)
    top = int(upper_ratio * box.h)
    m[:top, :] = 0.0
    k = max(3, int(blur_frac * box.w)) | 1  # odd kernel
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m[..., None]


def _unsharp(img: np.ndarray, amount: float = 0.4, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def paste_back(
    frame_bgr: np.ndarray, gen_face_bgr: np.ndarray, box: FaceBox, landmarks: np.ndarray
) -> np.ndarray:
    """Blend the generated lower face back into the frame via the jaw mask."""
    out = frame_bgr.copy()
    face = cv2.resize(
        _unsharp(gen_face_bgr), (box.w, box.h), interpolation=cv2.INTER_LANCZOS4
    )
    region = out[box.y1 : box.y2, box.x1 : box.x2].astype(np.float32)
    alpha = jaw_mask(landmarks, box)
    blended = face.astype(np.float32) * alpha + region * (1.0 - alpha)
    out[box.y1 : box.y2, box.x1 : box.x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
