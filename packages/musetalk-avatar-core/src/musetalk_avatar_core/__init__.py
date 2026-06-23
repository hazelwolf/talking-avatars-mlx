"""MuseTalk MLX avatar core — torch-free video-driven lip-sync engine.

Wraps the musetalk_mlx neural pipeline with landmark-based cropping and
jaw-contour blending (the caller-supplied pre/post-processing).
"""

from . import avatar, face_utils
from .avatar import DEFAULT_VIDEO, FPS, VideoAvatar, extract_audio, render_video

__all__ = [
    "avatar",
    "face_utils",
    "VideoAvatar",
    "render_video",
    "extract_audio",
    "DEFAULT_VIDEO",
    "FPS",
]
