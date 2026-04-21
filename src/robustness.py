"""Robustness control experiments (DriveBench-style).

Three controls per video:
  - text_only : no images, only annotation + pixel summary text
  - noise     : images replaced by pure random-noise frames
  - identical : real frames used as both 'real' and 'generated' -> expected 0

If the normal evaluation score is driven by language priors only, text_only
and noise should score similarly to the normal run; a good visual grounder
should make `identical` collapse to a near-zero score.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from .config import CFG
from .evaluator import evaluate_one_video
from .vlm_client import VLMClient

_FRAME_W, _FRAME_H = CFG["robustness"]["noise_frame_size"]


def _noise_frame(h: int = _FRAME_H, w: int = _FRAME_W) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def run_robustness_for_video(
    client: VLMClient,
    real_frames: Sequence[Image.Image],
    gen_frames: Sequence[Image.Image],
    pixel_metrics: list,
    temporal: dict,
    annotation: dict,
    video_id: str,
) -> Dict:
    # identical: feed real as generated -> should be extremely low unsafe score
    identical_result = evaluate_one_video(
        client, real_frames, real_frames, pixel_metrics, temporal,
        annotation, f"{video_id}::identical", num_samples=1,
    )

    # noise: 3 random-noise frames in the "generated" slots
    noise_gen = [_noise_frame() for _ in real_frames]
    noise_result = evaluate_one_video(
        client, real_frames, noise_gen, pixel_metrics, temporal,
        annotation, f"{video_id}::noise", num_samples=1,
    )

    # text-only: blank both top & bottom with identical black frames
    blank = Image.new("RGB", (_FRAME_W, _FRAME_H), (0, 0, 0))
    text_only_result = evaluate_one_video(
        client, [blank] * len(real_frames), [blank] * len(real_frames),
        pixel_metrics, temporal, annotation,
        f"{video_id}::text_only", num_samples=1,
    )

    return {
        "video_id": video_id,
        "identical": identical_result,
        "noise": noise_result,
        "text_only": text_only_result,
    }
