# musetalk-avatar-core

Torch-free, video-driven lip-sync engine for MuseTalk 1.5 (MLX q4) on Apple Silicon.

Wraps the `musetalk_mlx` neural pipeline with the caller-supplied pre/post-processing:
MediaPipe landmark cropping (`face_utils`) and a ping-pong frame-cycling driver with
EMA-smoothed crop boxes and jaw-contour blending (`avatar`).
