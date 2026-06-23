# Fixing the blurred lip region

**Date:** 2026-06-23
**Component:** `face_utils.py` (post-processing), not the model

## Symptom

Video-driven generation (`assets/first-cut.MP4` + MuseTalk-1.5 MLX q4) produced
a talking head whose **mouth/lip region was visibly blurred**, while the rest of the
face stayed sharp.

## Investigation

Diagnosed by isolating each stage rather than guessing.

1. **VAE round-trip (encode → decode, no UNet) was sharp.**
   Encoding the reference crop and decoding it straight back produced a crisp mouth.
   → The VAE and the crop resolution are not the problem.

2. **UNet output was soft only in the lower half.**
   Decoding the UNet's single-step inpainting result showed a sharp upper face
   (eyes/forehead, effectively passed through from the reference latent) but a
   **smooth lower half** (mouth, cheeks, jaw, neck — the region MuseTalk regenerates).
   → The softness lives in the generated region.

3. **Quantization was ruled out.**
   Downloaded the **q8** variant (config reports cosine `1.0` vs fp16, vs q4's
   `0.99985`) and generated the identical mouth strip. The mouth was **visually
   identical** to q4 — still soft.
   → Not the 4-bit weights; switching to q8/fp16 would not help. (q8 was removed
   again; q4 remains the default.)
   Note: cosine similarity is dominated by low frequencies, so a high score does not
   guarantee preserved high-frequency mouth detail — but here even q8 matched q4, so
   quantization was not the cause regardless.

4. **Audio conditioning was healthy.**
   Generating across several audio chunks (0, 10, 20, 28, 40, 55) produced clearly
   **different** mouth shapes (open/closed/teeth). The mouth tracks the audio
   correctly; frames were just uniformly soft, not averaged/dead.

**Conclusion:** the blur was caused by the **post-processing**, not the model:

- The paste-back blended the **entire regenerated lower half** (cheeks, jaw, neck —
  all smoothed by the model) over the sharp original skin. Only the mouth actually
  needs the generated pixels; everything else was being needlessly replaced with
  softer content.
- The face crop was too loose (`expand=0.5`), so the mouth occupied few pixels in the
  256×256 crop and lost detail when scaled back up to the 1024×576 frame.

## Fix

All in `face_utils.py`:

1. **Tighter crop** — `compute_face_box` defaults `expand 0.5 → 0.1`,
   `down_bias 0.15 → 0.0`. The face fills more of the 256² crop, so the mouth gets
   far more resolution.
2. **Mouth-region blend** — replaced the full lower-face alpha mask with a feathered
   **elliptical mask over the mouth only** (`_mouth_mask`). Original (sharp) pixels
   are kept everywhere except the mouth. This mirrors how upstream
   [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) blends a focused
   face-parsing mask rather than the whole crop.
3. **Light unsharp mask** — `_unsharp` (amount 0.6, sigma 1.2) applied to the
   generated face before blending, to counter the residual softness of generated
   pixels.

## Result

Defined lips and visible teeth, blended seamlessly into the original frame, verified
on a frame extracted from the final muxed video. The fix applies to both the live
stream and the rendered mp4.

## Trade-off

The mouth-region mask keeps the **original jaw**, so very large jaw movements show
slightly less than if the whole lower face were regenerated. Not noticeable on this
driving video. If maximum jaw motion is wanted (at the cost of some softness), expose
the mask size (`_mouth_mask` `ax`/`ay`) and `_unsharp` `amount` as UI controls.
