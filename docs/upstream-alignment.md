# Aligning the inference pipeline with upstream MuseTalk

**Date:** 2026-06-23
**Goal:** match the smoother, better-synced output of
[TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) without its CUDA /
torch / mmpose stack.

## Background

The neural core (VAE + UNet + Whisper) is identical to upstream — we use the MLX q4
port of the same GAN/perceptual/sync-trained weights. The quality gap was entirely
in the **caller-supplied** preprocessing/blending that upstream does and the
`musetalk_mlx` package leaves to us. Three differences mattered:

| Aspect | Upstream | Our previous code | Effect |
|---|---|---|---|
| Face crop | DWPose 68-pt landmarks → nose-centred box to chin, `bbox_shift`/`extra_margin` | OpenCV Haar box, square, fixed expand | Misaligned + jittery crop → softer, wobbly |
| Stability | Stable per-frame landmark boxes | Haar box jitters frame-to-frame | "Not smooth" |
| Blend | BiSeNet face-parse mask, lower half, blurred | Static elliptical mouth patch | Floating mouth, no jaw motion |

## Changes

All inference-side; no retraining (the sharpness/sync is already baked into the
weights by upstream's stage-2 GAN + perceptual + SyncNet losses).

### 1. Landmark-based crop (`face_utils.py`)
- Detection: **MediaPipe FaceLandmarker** (478 landmarks), torch-free, arm64 wheels,
  one-time 3.7 MB model download. Replaces the Haar cascade.
- Box: mirrors upstream `get_landmark_and_bbox` — x spans the face-oval width; the box
  is vertically centred on a nose reference (`NOSE_REF = 6`) and reaches the chin;
  `bbox_shift` nudges the nose reference (mouth-in-frame control) and `extra_margin`
  pads the chin (v1.5 convention). Crop resized to 256 with **LANCZOS4**.
- Picked `NOSE_REF` empirically by sweeping candidate nose-bridge indices and
  choosing the one that frames the face like the training crop (eyes upper third,
  mouth lower third, chin at the bottom).

### 2. Temporal stabilization (`avatar.py`)
- Crop-box coordinates are **EMA-smoothed** across the video's frame order
  (`smooth = 0.5`). Measured ~25% additional frame-to-frame jitter reduction on top
  of the large gain from landmarks (raw landmark jitter ≈ 1.4 px vs Haar's much
  larger wobble). Smoothing the box only affects crop position, not lip timing.

### 3. Jaw-contour blend (`face_utils.jaw_mask`)
- Replaced the static mouth ellipse with a feathered **convex hull of the face-oval
  landmarks**, lower half kept (`upper_ratio = 0.5`), Gaussian-blurred edges — the
  torch-free analogue of upstream's parse-mask blend. The jaw now moves with speech
  and the boundary follows the real face contour. Light unsharp retained (reduced to
  0.4) since the aligned crop is already sharper.

## Result

Open-mouth frames show defined lips and visible teeth at native resolution, natural
jaw motion, and seamless blending; box jitter is reduced. End-to-end throughput
~7 fps on first pass (q4, M-series), cached frames replay faster.

## Deliberately NOT done
- **No training.** Upstream's two-stage training (L1 + VGG perceptual + GAN +
  SyncNet sync loss, HDTF data, 250k steps, multi-GPU CUDA) is what produced the
  weights; we consume the MLX port and only fix preprocessing/blending.
- **No torch / mmpose / DWPose / BiSeNet.** All replaced with MediaPipe + OpenCV to
  stay Apple-Silicon-native and torch-free, consistent with `musetalk_mlx`.
- **fp16 weights** untried for quality — q8 was visually identical to q4, so the
  win was crop/blend, not precision. onnxruntime (CoreML) is available if a true
  BiSeNet parse mask is ever wanted.
