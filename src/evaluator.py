"""Score aggregation and high-level evaluate-one-video routine."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from .config import CFG
from .prompts import COMBINED_PROMPT, build_annotation_desc, build_pixel_summary
from .vlm_client import VLMClient


POISON_THRESHOLD: float = CFG["evaluation"]["poison_threshold"]
WEIGHTS: dict = CFG["evaluation"]["weights"]
KEY_FRAME_INDICES: tuple = tuple(CFG["evaluation"]["key_frame_indices"])


def compute_final_score(sem: float, log: float, dec: float) -> Dict:
    final = (
        WEIGHTS["semantic"] * sem
        + WEIGHTS["logical"] * log
        + WEIGHTS["decision"] * dec
    )
    is_poisoned = max(sem, log, dec) >= POISON_THRESHOLD
    if not is_poisoned:
        attack_level = "None"
    else:
        # pick the highest-scoring dimension; ties broken by priority: Decision > Semantic > Logical
        ranked = sorted(
            [("Decision", dec), ("Semantic", sem), ("Logical", log)],
            key=lambda x: x[1],
            reverse=True,
        )
        top_score = ranked[0][1]
        priority = {"Decision": 0, "Semantic": 1, "Logical": 2}
        candidates = [name for name, s in ranked if s == top_score]
        attack_level = min(candidates, key=lambda n: priority[n])
    return {
        "is_poisoned": is_poisoned,
        "attack_level": attack_level,
        "scores": {
            "semantic": round(sem, 2),
            "logical": round(log, 2),
            "decision": round(dec, 2),
        },
        "final_score": round(final, 2),
    }


def aggregate_samples(samples: List[Dict], video_id: str) -> Dict:
    """Median-fuse multiple sampling results for a single video."""
    sem = float(np.median([s["scores"]["semantic"] for s in samples]))
    log = float(np.median([s["scores"]["logical"] for s in samples]))
    dec = float(np.median([s["scores"]["decision"] for s in samples]))
    base = compute_final_score(sem, log, dec)
    mid = samples[len(samples) // 2]
    return {
        "video_id": video_id,
        **base,
        "reasoning": mid.get("reasoning", ""),
    }


def evaluate_one_video(
    client: VLMClient,
    real_frames: Sequence[Image.Image],
    gen_frames: Sequence[Image.Image],
    pixel_metrics: list,
    temporal: dict,
    annotation: dict,
    video_id: str,
    num_samples: int | None = None,
    frame_resize: tuple | None = None,
    temperature: float | None = None,
) -> Dict:
    """Run the COMBINED prompt `num_samples` times, then median-fuse."""
    if num_samples is None:
        num_samples = CFG["evaluation"]["num_samples"]
    if frame_resize is None:
        frame_resize = tuple(CFG["evaluation"]["frame_resize"])
    if temperature is None:
        temperature = CFG["inference"]["temperature"]

    annotation_desc = build_annotation_desc(annotation)
    pixel_summary = build_pixel_summary(pixel_metrics, temporal)

    prompt = COMBINED_PROMPT.format(
        annotation_desc=annotation_desc,
        pixel_summary=pixel_summary,
        video_id=video_id,
    )

    image_blocks = []
    n = len(real_frames)
    for idx in KEY_FRAME_INDICES:
        if idx >= n:
            continue
        ts = idx * 0.25
        real_img = real_frames[idx]
        gen_img = gen_frames[idx]
        if frame_resize is not None:
            real_img = real_img.resize(frame_resize, Image.LANCZOS)
            gen_img = gen_img.resize(frame_resize, Image.LANCZOS)
        image_blocks.append({"label": f"\n--- t={ts:.2f}s | TOP=real", "image": real_img})
        image_blocks.append({"label": "BOTTOM=generated:", "image": gen_img})

    samples: List[Dict] = []
    for i in range(num_samples):
        try:
            reply = client.chat_json(
                prompt=prompt,
                image_blocks=image_blocks,
                temperature=temperature,
                max_tokens=CFG["inference"]["max_tokens"],
                enable_thinking=CFG["inference"]["enable_thinking"],
            )
            reply.setdefault("scores", {})
            for k in ("semantic", "logical", "decision"):
                reply["scores"].setdefault(k, 0.0)
                reply["scores"][k] = float(reply["scores"][k])
            samples.append(reply)
        except Exception as e:  # noqa: BLE001
            print(f"  sample {i+1} failed: {e}")
    if not samples:
        return {
            "video_id": video_id,
            "is_poisoned": None,
            "attack_level": "Error",
            "scores": {"semantic": -1, "logical": -1, "decision": -1},
            "final_score": -1,
            "reasoning": "VLM evaluation failed for all samples",
        }

    return aggregate_samples(samples, video_id)
