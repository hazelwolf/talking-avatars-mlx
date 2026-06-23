"""Generate the shipped sample driving video: assets/sample_avatar.mp4.

Source face is a synthetic StyleGAN2 portrait (thispersondoesnotexist.com) — not a
real person, so there are no privacy/licensing concerns shipping it. We add subtle
head motion (so it behaves like a real driving video) and a short spoken track via
macOS `say` (so the "use the video's own audio" path is demonstrable too).

Usage:  uv run python scripts/make_sample_video.py
Re-run to regenerate. Requires `say` (macOS) and ffmpeg.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import urllib.request

import cv2
import numpy as np

OUT = "assets/sample_avatar.mp4"
SIZE = 512
FPS = 25
TEXT = (
    "Hi! I'm a sample avatar for MuseTalk M.L.X. "
    "Swap in your own video, or just type what you'd like me to say."
)
VOICE = os.environ.get("SAMPLE_VOICE", "Samantha")


def _source_face() -> np.ndarray:
    path = "samples/avatar.jpg"
    os.makedirs("samples", exist_ok=True)
    if not os.path.exists(path):
        req = urllib.request.Request(
            "https://thispersondoesnotexist.com", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            open(path, "wb").write(r.read())
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("Could not obtain a source face image.")
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _say_to_wav() -> tuple[str, float]:
    aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False).name
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["say", "-v", VOICE, "-o", aiff, TEXT], check=True, capture_output=True
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", wav],
        check=True,
        capture_output=True,
    )
    dur = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                wav,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    os.unlink(aiff)
    return wav, dur


def _frame(face: np.ndarray, t: float) -> np.ndarray:
    """Subtle head motion via a small affine warp (translate / rotate / breathe)."""
    dx = 6.0 * math.sin(2 * math.pi * 0.18 * t)
    dy = 4.0 * math.sin(2 * math.pi * 0.13 * t + 1.0)
    ang = 1.2 * math.sin(2 * math.pi * 0.10 * t)
    scale = 1.02 + 0.012 * math.sin(2 * math.pi * 0.22 * t)
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(face, m, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)


def main():
    os.makedirs("assets", exist_ok=True)
    face = _source_face()
    wav, dur = _say_to_wav()
    n = max(FPS, int(round((dur + 0.3) * FPS)))  # small tail of silence

    silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    for i in range(n):
        vw.write(_frame(face, i / FPS))
    vw.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            silent,
            "-i",
            wav,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-c:a",
            "aac",
            "-shortest",
            OUT,
        ],
        check=True,
        capture_output=True,
    )
    os.unlink(silent)
    print(f"wrote {OUT}  ({n} frames @ {FPS}fps, {dur:.1f}s audio)")


if __name__ == "__main__":
    main()
