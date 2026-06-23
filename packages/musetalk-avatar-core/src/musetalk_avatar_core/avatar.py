"""Video-driven avatar lip-sync, following the MuseTalk realtime-inference pattern
(https://github.com/TMElyralab/MuseTalk).

A driving video supplies natural head motion. Its frames are cycled with a
ping-pong order (0..N-1, N-2..1, ...) so playback loops seamlessly to whatever
audio length is requested. For each used frame we detect landmarks once, derive an
upstream-style crop box, cache its UNet latent, run single-step inpainting
conditioned on the audio, and blend the generated lower face back through a jaw mask.

Crop boxes are EMA-smoothed across the video's frame order to remove detection
jitter (the main source of non-smooth output). Frames are read lazily and cached.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator

import cv2
import mlx.core as mx
import numpy as np
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

from . import face_utils

MODEL_DIR = os.environ.get("MUSETALK_MODEL_DIR", "models/MuseTalk-1.5-q4")
# Prefer a local personal clip if present, else the shipped synthetic sample.
_DEFAULT_VIDEO = (
    "assets/first-cut.MP4"
    if os.path.exists("assets/first-cut.MP4")
    else "assets/sample_avatar.mp4"
)
DEFAULT_VIDEO = os.environ.get("MUSETALK_VIDEO", _DEFAULT_VIDEO)
FPS = 25

_pipe: MuseTalkPipeline | None = None


def get_pipeline() -> MuseTalkPipeline:
    """Load (once) and cache the MLX q4 pipeline."""
    global _pipe
    if _pipe is None:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_DIR!r}. Download with:\n"
                f"  uv run hf download mlx-community/MuseTalk-1.5-q4 "
                f"--local-dir {MODEL_DIR}"
            )
        _pipe = MuseTalkPipeline.from_pretrained_mlx(MODEL_DIR)
    return _pipe


def extract_audio(video_path: str) -> str:
    """Extract a 16kHz mono wav from a video's audio track."""
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", wav],
        check=True,
        capture_output=True,
    )
    return wav


def _cycle_index(step: int, n: int) -> int:
    """Map a monotonically increasing step to a ping-pong frame index in [0, n)."""
    if n <= 1:
        return 0
    period = 2 * n - 2
    pos = step % period
    return pos if pos < n else period - pos


class VideoAvatar:
    """Lazily-read driving video with cached, EMA-smoothed per-frame crop boxes,
    landmarks, and UNet latents."""

    def __init__(
        self,
        video_path: str,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        smooth: float = 0.5,
    ):
        self.path = video_path
        self.bbox_shift = bbox_shift
        self.extra_margin = extra_margin
        self.smooth = smooth
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frames: dict[int, np.ndarray] = {}
        self._boxes: dict[int, face_utils.FaceBox] = {}
        self._lms: dict[int, np.ndarray] = {}
        self._latents: dict[int, mx.array] = {}
        self._read_pos = 0
        self._last_lm: np.ndarray | None = None
        self._sbox: np.ndarray | None = None  # running EMA of [x1,y1,x2,y2]

    def _ensure_frame(self, idx: int):
        """Read sequentially until frame `idx` is cached, detecting + smoothing."""
        while idx not in self._frames and self._read_pos < self.n_frames:
            ok, frame = self.cap.read()
            i = self._read_pos
            self._read_pos += 1
            if not ok:
                self.n_frames = i
                break
            self._frames[i] = frame
            lm = face_utils.detect_landmarks(frame)
            if lm is None:
                lm = self._last_lm
            else:
                self._last_lm = lm
            self._lms[i] = lm
            if lm is None:
                self._boxes[i] = None
                continue
            raw = face_utils.box_from_landmarks(
                lm, frame.shape, self.bbox_shift, self.extra_margin
            )
            coords = np.array([raw.x1, raw.y1, raw.x2, raw.y2], dtype=np.float32)
            if self._sbox is None:
                self._sbox = coords
            else:
                self._sbox = self.smooth * coords + (1 - self.smooth) * self._sbox
            s = self._sbox.round().astype(int)
            self._boxes[i] = face_utils.FaceBox(
                int(s[0]), int(s[1]), int(s[2]), int(s[3])
            )
        if idx not in self._frames:
            raise IndexError(f"frame {idx} unavailable (n_frames={self.n_frames})")
        if self._boxes.get(idx) is None:
            raise ValueError("No face detected in the driving video.")

    def _latent(self, idx: int) -> mx.array:
        if idx not in self._latents:
            self._ensure_frame(idx)
            crop = face_utils.crop_face(self._frames[idx], self._boxes[idx])
            self._latents[idx] = get_pipeline().get_latents_for_unet(crop)
        return self._latents[idx]

    def stream_frames(self, wav_path: str, batch_size: int = 8) -> Iterator[np.ndarray]:
        """Yield lip-synced RGB frames for the given audio, cycling the video."""
        pipe = get_pipeline()
        chunks = pipe.encode_audio_from_wav(wav_path, fps=FPS)
        n = int(chunks.shape[0])
        if n == 0:
            return
        self._ensure_frame(0)
        order = [_cycle_index(s, self.n_frames) for s in range(n)]

        for i in range(0, n, batch_size):
            idxs = order[i : i + batch_size]
            latent_stack = mx.concatenate([self._latent(j) for j in idxs], axis=0)
            faces = pipe.run_batched(
                latent_stack, chunks[i : i + batch_size], batch_size=batch_size
            )
            for j, face in zip(idxs, faces, strict=False):
                frame = face_utils.paste_back(
                    self._frames[j], face, self._boxes[j], self._lms[j]
                )
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_video(
    video_path: str,
    wav_path: str,
    out_path: str,
    batch_size: int = 8,
    bbox_shift: int = 0,
    extra_margin: int = 10,
) -> str:
    """Render the full lip-synced clip to mp4 with the audio muxed in."""
    av = VideoAvatar(video_path, bbox_shift, extra_margin)
    h = int(av.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(av.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    try:
        wrote = False
        for frame_rgb in av.stream_frames(wav_path, batch_size):
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            wrote = True
    finally:
        writer.release()
    if not wrote:
        raise ValueError("Audio produced no frames.")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            tmp_video,
            "-i",
            wav_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    os.unlink(tmp_video)
    return out_path
