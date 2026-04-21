#!/usr/bin/env python3
"""Compute human-machine consistency metrics from VLM results and human annotations.

Usage:
  python scripts/run_compare.py \
      --dataset_json results/dataset.json \
      --annotation_xlsx human_evaluate/annotation.xlsx \
      --output_dir results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.compare_analysis import analyse
from src.utils import setup_logger

logger = setup_logger("run_compare")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json",
                   default="results/dataset.json")
    p.add_argument("--annotation_xlsx",
                   default="human_evaluate/annotation.xlsx")
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_json = os.path.join(ROOT, args.dataset_json) \
        if not os.path.isabs(args.dataset_json) else args.dataset_json
    annotation_xlsx = os.path.join(ROOT, args.annotation_xlsx) \
        if not os.path.isabs(args.annotation_xlsx) else args.annotation_xlsx
    output_dir = os.path.join(ROOT, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir

    if not os.path.exists(dataset_json):
        logger.error("dataset.json not found: %s", dataset_json)
        sys.exit(1)
    if not os.path.exists(annotation_xlsx):
        logger.error("annotation.xlsx not found: %s", annotation_xlsx)
        sys.exit(1)

    logger.info("Loading VLM results from %s", dataset_json)
    logger.info("Loading human annotations from %s", annotation_xlsx)

    metrics = analyse(dataset_json, annotation_xlsx)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "comparison_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved -> %s", out_path)

    # pretty print summary
    print("\n" + "=" * 55)
    print("Human-Machine Consistency Summary")
    print("=" * 55)
    for dim, d in metrics["dimensions"].items():
        print(f"  {dim:10s}  Spearman={d['spearman_r']:+.3f}  "
              f"Kappa={d['weighted_kappa']:+.3f}  "
              f"ICC={d['icc21_human_vs_vlm']:+.3f}")
    ov = metrics["overall"]
    print(f"  {'overall':10s}  Spearman={ov['spearman_r']:+.3f}  "
          f"ICC(final)={ov['icc21_final_score']:+.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
