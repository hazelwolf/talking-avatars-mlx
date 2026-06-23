#!/usr/bin/env bash
# Setup for the MuseTalk MLX talking-avatar demo (Apple Silicon).
# Idempotent: safe to re-run. Creates the uv env and pre-fetches every model the
# app downloads at runtime, so the first generation is fast and works offline.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${MUSETALK_MODEL_DIR:-models/MuseTalk-1.5-q4}"
HF_REPO="mlx-community/MuseTalk-1.5-q4"
LANDMARKER="${MUSETALK_LANDMARKER:-models/face_landmarker.task}"
LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

echo "==> Checking prerequisites"
command -v uv >/dev/null     || { echo "ERROR: 'uv' not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: 'ffmpeg' not found — needed for audio/video muxing and TTS (brew install ffmpeg)."

echo "==> Syncing Python environment (uv sync)"
uv sync

# Install the git pre-commit hook (ruff lint + format) when in a git repo.
if [ -d .git ]; then
  echo "==> Installing pre-commit hook"
  uv run pre-commit install >/dev/null
fi

echo "==> Downloading MuseTalk q4 weights -> $MODEL_DIR"
if [ -f "$MODEL_DIR/unet.safetensors" ] && [ -f "$MODEL_DIR/vae.safetensors" ] \
   && [ -f "$MODEL_DIR/whisper_encoder.safetensors" ]; then
  echo "    already present, skipping."
else
  uv run hf download "$HF_REPO" --local-dir "$MODEL_DIR"
fi

echo "==> Downloading MediaPipe face landmarker -> $LANDMARKER"
if [ -f "$LANDMARKER" ]; then
  echo "    already present, skipping."
else
  mkdir -p "$(dirname "$LANDMARKER")"
  curl -fL --retry 3 -o "$LANDMARKER" "$LANDMARKER_URL"
fi

echo
echo "==> Done. Run the app with:"
echo "      uv run musetalk-avatar    # then open http://127.0.0.1:7860"
