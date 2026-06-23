"""Minimal Gradio interface for MuseTalk-1.5 (MLX, q4) video-driven talking avatar.

A driving video (default: assets/sample_avatar.mp4) provides natural head motion;
its frames are cycled to match the audio. Make it talk with:
  1. Typed text  -> macOS `say` synthesizes speech (offline).
  2. An uploaded / recorded audio clip.
  3. The driving video's own audio (re-lip-sync to itself).

Frames stream live into the UI as MLX generates them; a downloadable mp4 (with
audio muxed) is rendered at the end. Follows the MuseTalk realtime-inference design
(https://github.com/TMElyralab/MuseTalk).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import gradio as gr

from musetalk_avatar_core import avatar

SAMPLE_RATE = 16000
DEFAULT_VIDEO = avatar.DEFAULT_VIDEO


def _text_to_wav(text: str, voice: str | None = None) -> str:
    if not shutil.which("say"):
        raise gr.Error("macOS `say` not available; upload an audio file instead.")
    aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False).name
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = ["say", "-o", aiff]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    subprocess.run(cmd, check=True, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", aiff, "-ar", str(SAMPLE_RATE), "-ac", "1", wav],
        check=True,
        capture_output=True,
    )
    os.unlink(aiff)
    return wav


def _resolve_audio(video_path, audio_path, text, voice, use_video_audio):
    if use_video_audio:
        return avatar.extract_audio(video_path)
    if audio_path:
        return audio_path
    if text and text.strip():
        return _text_to_wav(text.strip(), voice or None)
    raise gr.Error("Provide text, an audio clip, or tick 'use video's own audio'.")


def generate(
    video_path,
    audio_path,
    text,
    voice,
    use_video_audio,
    batch_size,
    bbox_shift,
    extra_margin,
):
    if not video_path:
        raise gr.Error("Select a driving video first.")
    wav_path = _resolve_audio(video_path, audio_path, text, voice, use_video_audio)

    av = avatar.VideoAvatar(video_path, int(bbox_shift), int(extra_margin))
    last = None
    for frame_rgb in av.stream_frames(wav_path, int(batch_size)):
        last = frame_rgb
        yield frame_rgb, gr.update()

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    avatar.render_video(
        video_path,
        wav_path,
        out_path,
        int(batch_size),
        int(bbox_shift),
        int(extra_margin),
    )
    yield last, gr.update(value=out_path, visible=True)


with gr.Blocks(title="MuseTalk MLX — Video Avatar") as demo:
    gr.Markdown(
        "# 🎬 MuseTalk 1.5 (MLX q4) — Video-Driven Talking Avatar\n"
        "A driving video provides head motion; the mouth is re-synced to your audio. "
        "Frames stream live; a video with sound is rendered at the end."
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(
                label="Driving video",
                height=300,
                value=DEFAULT_VIDEO if os.path.exists(DEFAULT_VIDEO) else None,
            )
            with gr.Tab("Type text"):
                text_in = gr.Textbox(
                    label="Text to speak",
                    lines=3,
                    placeholder="Hello! I am a MuseTalk MLX avatar.",
                )
                voice_in = gr.Textbox(
                    label="macOS voice (optional)", placeholder="e.g. Samantha, Daniel"
                )
            with gr.Tab("Upload audio"):
                audio_in = gr.Audio(
                    label="Audio", type="filepath", sources=["upload", "microphone"]
                )
            use_video_audio = gr.Checkbox(
                label="Use the driving video's own audio", value=False
            )
            with gr.Accordion("Advanced (crop tuning)", open=False):
                bbox_shift_in = gr.Slider(
                    -30,
                    30,
                    value=0,
                    step=1,
                    label="bbox_shift (↓ shows more mouth, ↑ less)",
                )
                extra_margin_in = gr.Slider(
                    0, 40, value=10, step=1, label="extra_margin (chin padding, px)"
                )
            batch_in = gr.Slider(1, 16, value=8, step=1, label="Batch size")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            live = gr.Image(label="Live preview (no sound)", height=300)
            video_out = gr.Video(label="Result (with sound)", visible=False)

    go.click(
        generate,
        inputs=[
            video_in,
            audio_in,
            text_in,
            voice_in,
            use_video_audio,
            batch_in,
            bbox_shift_in,
            extra_margin_in,
        ],
        outputs=[live, video_out],
    )


def main():
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
