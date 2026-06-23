# MuseTalk 1.5 (MLX q4) — Video-Driven Talking Avatar

A minimal Gradio interface for video-driven realtime lip-sync on Apple Silicon,
built on the [mlx-community/MuseTalk-1.5-q4](https://huggingface.co/mlx-community/MuseTalk-1.5-q4)
weights and the [xocialize/musetalk-mlx](https://github.com/xocialize/musetalk-mlx)
inference package. It follows the realtime-inference design of the original
[TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk).

A **driving video** (default) supplies natural head
motion; its frames are cycled to match your audio while the mouth is re-synced.
Drive it with typed text (macOS `say`), an uploaded/recorded clip, or the video's
own audio. Frames stream live, then a downloadable mp4 with sound is rendered.

## How it works

The `musetalk_mlx` package owns only the neural path (VAE encode → single-step UNet
inpainting conditioned on Whisper audio features → VAE decode). Face detection,
256×256 cropping, frame cycling, and paste-back are caller-supplied, so this repo adds:

- `musetalk_avatar_core.face_utils` — **MediaPipe FaceLandmarker** (478 landmarks,
  torch-free) for detection; an upstream-style crop box (face width ×
  nose-centred-to-chin) with tunable `bbox_shift` and `extra_margin`; a 256² LANCZOS4
  crop; and paste-back via a feathered **jaw convex-hull mask** of the lower face (the
  torch-free analogue of upstream's BiSeNet parse-mask blend — the jaw moves
  naturally, edges stay smooth).
- `musetalk_avatar_core.avatar` — `VideoAvatar`: lazily reads the driving video,
  caches per-frame landmarks + **EMA-smoothed** crop boxes + UNet latents, and cycles
  frames **ping-pong** (0..N-1, N-2..1, …) so any audio length loops seamlessly.
  Exposes a streaming generator + ffmpeg-muxed render.
- `musetalk_avatar_app.app` — the Gradio UI.

## Workspace layout

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two
members sharing one `uv.lock` and one `.venv`:

```
pyproject.toml                     # virtual root: workspace + shared ruff config
packages/
  musetalk-avatar-core/            # torch-free lip-sync engine (avatar, face_utils)
  musetalk-avatar-app/             # gradio UI; exposes the `musetalk-avatar` command
```

Lint/format the whole tree with `uv run ruff check .` and `uv run ruff format .`
(one shared config at the root); a pre-commit hook runs both on commit.

## Setup (Apple Silicon, Python 3.12)

Managed entirely with [uv](https://docs.astral.sh/uv/) — `pyproject.toml` + `uv.lock`
pin a known-good resolution (the `numpy<2` / `numba>=0.60` constraints keep the
resolver off an ancient numba that has no Python 3.12 wheel).

One command does everything (env + all model downloads, idempotent):

```bash
./setup.sh
```

It runs `uv sync`, then fetches the q4 weights (~1.5 GB → `models/MuseTalk-1.5-q4`)
and the MediaPipe `face_landmarker.task` (3.7 MB → `models/`). Or do it manually:

```bash
uv sync
uv run hf download mlx-community/MuseTalk-1.5-q4 --local-dir models/MuseTalk-1.5-q4
```
(the landmarker model is auto-fetched on first run if missing.)

## Run

```bash
uv run musetalk-avatar
# open http://127.0.0.1:7860
```

Set `MUSETALK_MODEL_DIR` to point elsewhere for the weights.

## Sample driving video

A ready-to-use clip ships at `assets/sample_avatar.mp4` (a synthetic StyleGAN2 face
with subtle head motion + a spoken intro — no real person, freely shippable) and is
the default when no personal clip is present. The app prefers `assets/David_first-cut.MP4`
if you have it locally; otherwise it falls back to the sample. Regenerate the sample
with `uv run python scripts/make_sample_video.py`.

## Notes

- Driving video → talking head re-synced to new audio. Output is 25 fps (matches the
  asset); generation measured ~6 fps end-to-end on first pass (q4, M-series) — the
  first pass also reads frames + detects faces; cached frames replay faster.
- macOS `say` powers the text-to-speech tab; the audio tab accepts upload or mic;
  the checkbox re-lip-syncs the video to its own audio.
- Point `MUSETALK_VIDEO` / `MUSETALK_MODEL_DIR` elsewhere to change the defaults.
- "Live streaming" here means frames pushed to the UI as they generate. True
  webcam/mic round-trip conversation (STT + LLM + TTS) is out of scope for this
  minimal demo but would slot in on top of `VideoAvatar.stream_frames`.
- Face detection is MediaPipe FaceLandmarker; it needs a one-time 3.7 MB
  `face_landmarker.task` download (auto-fetched to `models/`, overridable via
  `MUSETALK_LANDMARKER`).
- `bbox_shift` / `extra_margin` (Advanced panel) mirror upstream's crop knobs — nudge
  them if the mouth framing looks off on a different driving video.
- See [docs/](docs/) for the lip-blur fix and the upstream-alignment changes.
```
