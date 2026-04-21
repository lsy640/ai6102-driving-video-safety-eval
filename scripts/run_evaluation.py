#!/usr/bin/env python3
"""End-to-end driving video safety evaluation.

Expects a vLLM server serving Qwen3.6-35B-A3B-FP8 at --port (default 8000).
Iterates the dataset, runs preprocessing + VLM scoring, writes dataset.json.

Usage:
  python scripts/run_evaluation.py \
      --video_dir VLM/dataset \
      --output_dir VLM/results \
      --port 8000 --num_samples 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make "src" importable when running as `python scripts/run_evaluation.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.preprocess import DrivingVideoProcessor
from src.vlm_client import VLMClient, DEFAULT_MODEL
from src.evaluator import evaluate_one_video
from src.utils import setup_logger


logger = setup_logger("run_eval")


def parse_args():
    _p = CFG["paths"]
    _v = CFG["vllm"]
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default=str(Path(_p["project_root"]) / _p["dataset"]))
    p.add_argument("--output_dir", default=str(Path(_p["project_root"]) / _p["results"]))
    p.add_argument("--port", type=int, default=_v["port"])
    p.add_argument("--host", default=_v["host"])
    p.add_argument("--model", default=_v["model"])
    p.add_argument("--num_samples", type=int, default=CFG["evaluation"]["num_samples"])
    p.add_argument("--limit", type=int, default=-1,
                   help="evaluate only first N videos (<=0 means all)")
    p.add_argument("--output_name", default="dataset.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    client = VLMClient(
        base_url=f"http://{args.host}:{args.port}/v1",
        model=args.model,
    )
    try:
        served = client.ping()
        logger.info("Connected to vLLM, served model: %s", served)
    except Exception as e:  # noqa: BLE001
        logger.error("Cannot reach vLLM at %s:%d -- %s",
                     args.host, args.port, e)
        sys.exit(1)

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if args.limit > 0:
        videos = videos[: args.limit]
    logger.info("Evaluating %d videos", len(videos))

    results = []
    t0 = time.time()
    for vi, vf in enumerate(videos, 1):
        vpath = os.path.join(args.video_dir, vf)
        logger.info("[%d/%d] %s", vi, len(videos), vf)

        try:
            proc = DrivingVideoProcessor(vpath)
            data = proc.extract_all_frames()
        except Exception as e:  # noqa: BLE001
            logger.error("  preprocess failed: %s", e)
            results.append({
                "video_id": vf,
                "is_poisoned": None,
                "attack_level": "Error",
                "scores": {"semantic": -1, "logical": -1, "decision": -1},
                "final_score": -1,
                "reasoning": f"preprocess error: {e}",
            })
            continue

        pixel_metrics = proc.compute_pixel_metrics(data)
        temporal = proc.compute_temporal_metrics(pixel_metrics)
        annotation = proc.parse_annotation_layer(data.frames[0].annotation)

        real_frames = [Image.fromarray(fd.real) for fd in data.frames]
        gen_frames = [Image.fromarray(fd.generated) for fd in data.frames]

        t_v0 = time.time()
        result = evaluate_one_video(
            client=client,
            real_frames=real_frames,
            gen_frames=gen_frames,
            pixel_metrics=pixel_metrics,
            temporal=temporal,
            annotation=annotation,
            video_id=vf,
            num_samples=args.num_samples,
        )
        dt = time.time() - t_v0

        result["automated_evaluation"] = {
            "model": args.model,
            "num_samples": args.num_samples,
            "pixel_metrics": {
                "per_frame_mae": [m["mae"] for m in pixel_metrics],
                "final_diff_pct": pixel_metrics[-1]["diff_area_pct"],
                "avg_psnr": round(float(np.mean([m["psnr"] for m in pixel_metrics])), 2),
                "temporal_slope": temporal["mae_slope"],
            },
            "seconds": round(dt, 2),
        }
        result["annotation_layer"] = annotation
        result["evaluation_criteria"] = "3-axis combined prompt v2.0 (English, anti-inflation)"
        result["prompt_version"] = "combined_v2_en"

        logger.info("  poisoned=%s level=%s final=%s (%.1fs)",
                    result.get("is_poisoned"),
                    result.get("attack_level"),
                    result.get("final_score"), dt)
        results.append(result)

        # incremental save every 5 videos
        if vi % 5 == 0:
            tmp = os.path.join(args.output_dir, f".{args.output_name}.partial")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    submission = [
        {
            "video_id": r["video_id"],
            "is_poisoned": r["is_poisoned"],
            "attack_level": r["attack_level"],
            "scores": r["scores"],
            "final_score": r["final_score"],
            "reasoning": r["reasoning"],
        }
        for r in results
    ]
    submission_path = os.path.join(
        args.output_dir, args.output_name.replace(".json", "_submission.json")
    )
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    poisoned = sum(1 for r in results if r.get("is_poisoned") is True)
    errors = sum(1 for r in results if r.get("attack_level") == "Error")
    logger.info("=" * 60)
    logger.info("Done %d videos in %.1fs", len(results), time.time() - t0)
    logger.info("poisoned=%d (%.1f%%) errors=%d",
                poisoned,
                poisoned / max(len(results), 1) * 100, errors)
    logger.info("results -> %s", out_path)
    logger.info("submission -> %s", submission_path)


if __name__ == "__main__":
    main()
