#!/usr/bin/env python3
"""Robustness control experiments on a small video subset."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.preprocess import DrivingVideoProcessor
from src.vlm_client import VLMClient, DEFAULT_MODEL
from src.robustness import run_robustness_for_video
from src.utils import setup_logger


logger = setup_logger("run_robustness")


def parse_args():
    _p = CFG["paths"]
    _v = CFG["vllm"]
    _r = CFG["robustness"]
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default=str(Path(_p["project_root"]) / _p["dataset"]))
    p.add_argument("--output_dir", default=str(Path(_p["project_root"]) / _p["results"]))
    p.add_argument("--port", type=int, default=_v["port"])
    p.add_argument("--host", default=_v["host"])
    p.add_argument("--model", default=_v["model"])
    p.add_argument("--num_videos", type=int, default=_r["num_videos"])
    p.add_argument("--seed", type=int, default=_r["seed"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    client = VLMClient(
        base_url=f"http://{args.host}:{args.port}/v1",
        model=args.model,
    )
    logger.info("Connected to vLLM, served model: %s", client.ping())

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    rng = random.Random(args.seed)
    sample = rng.sample(videos, min(args.num_videos, len(videos)))
    logger.info("Selected %d videos for robustness test", len(sample))

    out = []
    for vi, vf in enumerate(sample, 1):
        logger.info("[%d/%d] %s", vi, len(sample), vf)
        proc = DrivingVideoProcessor(os.path.join(args.video_dir, vf))
        data = proc.extract_all_frames()
        pixel_metrics = proc.compute_pixel_metrics(data)
        temporal = proc.compute_temporal_metrics(pixel_metrics)
        annotation = proc.parse_annotation_layer(data.frames[0].annotation)
        real = [Image.fromarray(fd.real) for fd in data.frames]
        gen = [Image.fromarray(fd.generated) for fd in data.frames]

        r = run_robustness_for_video(
            client, real, gen, pixel_metrics, temporal, annotation, vf,
        )
        out.append(r)
        logger.info("  identical=%.2f noise=%.2f text_only=%.2f",
                    r["identical"].get("final_score", -1),
                    r["noise"].get("final_score", -1),
                    r["text_only"].get("final_score", -1))

    out_path = os.path.join(args.output_dir, "robustness_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("saved -> %s", out_path)


if __name__ == "__main__":
    main()
