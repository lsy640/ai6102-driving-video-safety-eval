#!/usr/bin/env python3
"""Generate human–machine consistency figures.

Usage:
  python scripts/run_plot.py [--out_dir results/figures]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.plot_consistency import generate_all_figures


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate publication-quality consistency figures")
    p.add_argument("--dataset_json", default="results/dataset.json")
    p.add_argument("--annotation_xlsx", default="human_evaluate/annotation.xlsx")
    p.add_argument("--metrics_json", default="results/comparison_metrics.json")
    p.add_argument("--out_dir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    dataset_json = resolve(args.dataset_json)
    annotation_xlsx = resolve(args.annotation_xlsx)
    metrics_json = resolve(args.metrics_json)
    out_dir = resolve(args.out_dir)

    for path, label in [
        (dataset_json, "dataset.json"),
        (annotation_xlsx, "annotation.xlsx"),
        (metrics_json, "comparison_metrics.json"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    generate_all_figures(dataset_json, annotation_xlsx, metrics_json, out_dir)


if __name__ == "__main__":
    main()
