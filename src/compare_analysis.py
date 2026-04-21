"""Human-machine consistency analysis.

Computes per-dimension and overall Spearman correlation, weighted Cohen's Kappa,
and ICC(2,1) between the VLM-predicted scores and three human annotators.

Input
-----
- dataset.json  : VLM evaluation output (100 videos, fields: video_id, scores)
- annotation.xlsx: human annotations, columns:
    video_index, semantic1..3, logical1..3, decision1..3

Output
------
- results/comparison_metrics.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Kappa helpers ────────────────────────────────────────────────────────────

def _weighted_kappa(y1: np.ndarray, y2: np.ndarray,
                    levels: np.ndarray) -> float:
    """Linear-weighted Cohen's Kappa for ordinal scores."""
    idx = {v: i for i, v in enumerate(levels)}
    n = len(levels)
    k = len(y1)
    conf = np.zeros((n, n))
    for a, b in zip(y1, y2):
        conf[idx[round(a, 2)]][idx[round(b, 2)]] += 1
    conf /= k
    weights = 1 - np.abs(
        np.subtract.outer(np.arange(n), np.arange(n)) / (n - 1)
    )
    exp = np.outer(conf.sum(axis=1), conf.sum(axis=0))
    po = (conf * weights).sum()
    pe = (exp * weights).sum()
    return float((po - pe) / (1 - pe + 1e-12))


# ── ICC(2,1) ─────────────────────────────────────────────────────────────────

def _icc21(ratings: np.ndarray) -> float:
    """ICC(2,1) — two-way random effects, single measures, absolute agreement.

    ratings: shape (n_subjects, n_raters)
    """
    n, k = ratings.shape
    grand = ratings.mean()
    ssb = k * ((ratings.mean(axis=1) - grand) ** 2).sum()    # between subjects
    ssw = ((ratings - ratings.mean(axis=1, keepdims=True)) ** 2).sum()
    ssr = n * ((ratings.mean(axis=0) - grand) ** 2).sum()     # between raters
    sse = ssw - ssr
    msb = ssb / (n - 1)
    mse = sse / ((n - 1) * (k - 1))
    msr = ssr / (k - 1)
    icc = (msb - mse) / (msb + (k - 1) * mse + k * (msr - mse) / n)
    return float(icc)


# ── Main analysis ─────────────────────────────────────────────────────────────

LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
DIMS = ("semantic", "logical", "decision")
WEIGHTS = {"semantic": 0.3, "logical": 0.3, "decision": 0.4}


def load_vlm_scores(dataset_json: str) -> pd.DataFrame:
    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        vid = item["video_id"]
        idx = int(os.path.splitext(vid)[0])
        s = item.get("scores", {})
        rows.append({
            "video_index": idx,
            "vlm_semantic": s.get("semantic", np.nan),
            "vlm_logical":  s.get("logical", np.nan),
            "vlm_decision": s.get("decision", np.nan),
        })
    return pd.DataFrame(rows).sort_values("video_index").reset_index(drop=True)


def load_human_scores(annotation_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(annotation_xlsx)
    if "Unnamed: 0" in df.columns and "video_index" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "video_index"})
    return df.sort_values("video_index").reset_index(drop=True)


def analyse(dataset_json: str, annotation_xlsx: str) -> Dict:
    vlm = load_vlm_scores(dataset_json)
    human = load_human_scores(annotation_xlsx)

    merged = pd.merge(vlm, human, on="video_index", how="inner")
    n = len(merged)
    print(f"Matched {n} videos for human-machine comparison.")

    # human average score per dimension
    results: Dict = {"n_videos": n, "dimensions": {}, "overall": {}}

    for dim in DIMS:
        h_cols = [f"{dim}{i}" for i in (1, 2, 3)]
        h_mat = merged[h_cols].values.astype(float)   # (n, 3)
        h_avg = np.nanmean(h_mat, axis=1)
        vlm_s = merged[f"vlm_{dim}"].values.astype(float)

        # drop rows where either side is NaN
        valid = ~(np.isnan(h_avg) | np.isnan(vlm_s))
        h_v, m_v = h_avg[valid], vlm_s[valid]
        h_mat_v = h_mat[valid]

        spear_r, spear_p = stats.spearmanr(h_v, m_v)

        # snap to nearest level for kappa
        def snap(arr):
            return np.array([LEVELS[np.argmin(np.abs(LEVELS - x))] for x in arr])

        kappa = _weighted_kappa(snap(h_v), snap(m_v), LEVELS)

        # ICC: raters = [human_avg, vlm] — 2 raters (n, 2)
        icc_mat = np.stack([h_v, m_v], axis=1)
        icc = _icc21(icc_mat)

        # inter-human ICC for reference (n, 3)
        valid3 = ~np.any(np.isnan(h_mat_v), axis=1)
        icc_human = _icc21(h_mat_v[valid3]) if valid3.sum() > 2 else np.nan

        results["dimensions"][dim] = {
            "n_valid": int(valid.sum()),
            "human_avg_mean": round(float(h_v.mean()), 3),
            "vlm_mean": round(float(m_v.mean()), 3),
            "spearman_r": round(float(spear_r), 3),
            "spearman_p": round(float(spear_p), 4),
            "weighted_kappa": round(kappa, 3),
            "icc21_human_vs_vlm": round(icc, 3),
            "icc21_inter_human": round(float(icc_human), 3)
                if not np.isnan(icc_human) else None,
        }
        print(f"  [{dim}] Spearman={spear_r:.3f}(p={spear_p:.4f}) "
              f"Kappa={kappa:.3f} ICC={icc:.3f}")

    # overall final_score consistency
    h_final = sum(
        merged[[f"{d}{i}" for i in (1, 2, 3)]].mean(axis=1) * w
        for d, w in WEIGHTS.items()
    )
    vlm_final = (
        merged["vlm_semantic"] * WEIGHTS["semantic"]
        + merged["vlm_logical"] * WEIGHTS["logical"]
        + merged["vlm_decision"] * WEIGHTS["decision"]
    )
    valid_all = ~(h_final.isna() | vlm_final.isna())
    hf, mf = h_final[valid_all].values, vlm_final[valid_all].values
    spear_r, spear_p = stats.spearmanr(hf, mf)
    icc_overall = _icc21(np.stack([hf, mf], axis=1))

    results["overall"] = {
        "n_valid": int(valid_all.sum()),
        "spearman_r": round(float(spear_r), 3),
        "spearman_p": round(float(spear_p), 4),
        "icc21_final_score": round(icc_overall, 3),
    }
    print(f"  [overall] Spearman={spear_r:.3f} ICC={icc_overall:.3f}")
    return results
