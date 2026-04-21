"""Generate publication-quality human–machine consistency figures.

Produces 6 figures + 1 summary table for academic papers:
  1. Scatter plots with regression (per-dimension + overall)
  2. Bland–Altman agreement plots
  3. Confusion matrices (discretized scores)
  4. Radar chart of agreement metrics
  5. Score distribution comparison (violin + strip)
  6. Per-video deviation bar chart
  7. Summary statistics table (saved as image)

All figures follow Nature/IEEE style: serif fonts, 300 dpi, tight layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

from .compare_analysis import (
    DIMS, LEVELS, WEIGHTS,
    load_vlm_scores, load_human_scores,
    _icc21, _weighted_kappa,
)

# ── Global style ─────────────────────────────────────────────────────────────

FONT_FAMILY = "serif"
FONT_SIZE = 9
TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8

DIM_LABELS = {"semantic": "Semantic", "logical": "Logical", "decision": "Decision"}
DIM_COLORS = {"semantic": "#2166ac", "logical": "#4dac26", "decision": "#d6604d"}
DIM_MARKERS = {"semantic": "o", "logical": "s", "decision": "^"}

FIG_DPI = 300


def _apply_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _save(fig, path: str):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Data loading helper ──────────────────────────────────────────────────────

def _prepare_data(
    dataset_json: str, annotation_xlsx: str
) -> Tuple[pd.DataFrame, Dict]:
    vlm = load_vlm_scores(dataset_json)
    human = load_human_scores(annotation_xlsx)
    merged = pd.merge(vlm, human, on="video_index", how="inner")

    data = {}
    for dim in DIMS:
        h_cols = [f"{dim}{i}" for i in (1, 2, 3)]
        h_avg = merged[h_cols].mean(axis=1).values
        vlm_s = merged[f"vlm_{dim}"].values
        data[dim] = {"human": h_avg, "vlm": vlm_s, "h_mat": merged[h_cols].values}

    h_final = sum(
        merged[[f"{d}{i}" for i in (1, 2, 3)]].mean(axis=1) * w
        for d, w in WEIGHTS.items()
    )
    vlm_final = sum(merged[f"vlm_{d}"] * w for d, w in WEIGHTS.items())
    data["overall"] = {"human": h_final.values, "vlm": vlm_final.values}

    return merged, data


# ── Fig 1: Scatter + regression ──────────────────────────────────────────────

def plot_scatter_regression(data: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    items = list(DIMS) + ["overall"]
    for ax, key in zip(axes, items):
        h, v = data[key]["human"], data[key]["vlm"]
        color = DIM_COLORS.get(key, "#333333")
        marker = DIM_MARKERS.get(key, "D")
        label = DIM_LABELS.get(key, "Overall")

        ax.scatter(h, v, s=18, alpha=0.55, c=color, marker=marker,
                   edgecolors="white", linewidths=0.3, zorder=3)

        # regression line
        mask = ~(np.isnan(h) | np.isnan(v))
        if mask.sum() > 2:
            slope, intercept, r, p, se = stats.linregress(h[mask], v[mask])
            x_fit = np.linspace(0, 1, 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=color,
                    linewidth=1.2, linestyle="--", alpha=0.8)
            rho, p_rho = stats.spearmanr(h[mask], v[mask])
            ax.text(0.05, 0.92,
                    f"$\\rho$={rho:.3f}\n$R^2$={r**2:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.8))

        ax.plot([0, 1], [0, 1], "k:", linewidth=0.6, alpha=0.4)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Human Mean Score")
        ax.set_ylabel("VLM Score")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    fig.suptitle("Human vs. VLM Score Correlation", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig1_scatter_regression.pdf"))
    _save(fig if not plt.fignum_exists(fig.number) else
          _recreate_scatter(data), os.path.join(out_dir, "fig1_scatter_regression.png"))


def _recreate_scatter(data):
    """Re-create for PNG since PDF save closes the figure."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    items = list(DIMS) + ["overall"]
    for ax, key in zip(axes, items):
        h, v = data[key]["human"], data[key]["vlm"]
        color = DIM_COLORS.get(key, "#333333")
        marker = DIM_MARKERS.get(key, "D")
        label = DIM_LABELS.get(key, "Overall")
        ax.scatter(h, v, s=18, alpha=0.55, c=color, marker=marker,
                   edgecolors="white", linewidths=0.3, zorder=3)
        mask = ~(np.isnan(h) | np.isnan(v))
        if mask.sum() > 2:
            slope, intercept, r, p, se = stats.linregress(h[mask], v[mask])
            x_fit = np.linspace(0, 1, 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=color,
                    linewidth=1.2, linestyle="--", alpha=0.8)
            rho, _ = stats.spearmanr(h[mask], v[mask])
            ax.text(0.05, 0.92, f"$\\rho$={rho:.3f}\n$R^2$={r**2:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.8))
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.6, alpha=0.4)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Human Mean Score"); ax.set_ylabel("VLM Score")
        ax.set_title(label); ax.set_aspect("equal")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    fig.suptitle("Human vs. VLM Score Correlation", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ── Fig 2: Bland–Altman ──────────────────────────────────────────────────────

def plot_bland_altman(data: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, dim in zip(axes, DIMS):
        h, v = data[dim]["human"], data[dim]["vlm"]
        mean_val = (h + v) / 2
        diff = v - h
        md = np.nanmean(diff)
        sd = np.nanstd(diff, ddof=1)

        color = DIM_COLORS[dim]
        ax.scatter(mean_val, diff, s=16, alpha=0.5, c=color,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.axhline(md, color="k", linewidth=0.8, linestyle="-", label=f"Mean={md:.3f}")
        ax.axhline(md + 1.96 * sd, color="red", linewidth=0.7, linestyle="--",
                   label=f"+1.96SD={md+1.96*sd:.3f}")
        ax.axhline(md - 1.96 * sd, color="red", linewidth=0.7, linestyle="--",
                   label=f"−1.96SD={md-1.96*sd:.3f}")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")

        ax.set_xlabel("Mean of Human & VLM")
        ax.set_ylabel("Difference (VLM − Human)")
        ax.set_title(f"{DIM_LABELS[dim]}")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Bland–Altman Agreement Plots", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig2_bland_altman.pdf"))
    _save(_recreate_bland_altman(data),
          os.path.join(out_dir, "fig2_bland_altman.png"))


def _recreate_bland_altman(data):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, dim in zip(axes, DIMS):
        h, v = data[dim]["human"], data[dim]["vlm"]
        mean_val = (h + v) / 2
        diff = v - h
        md, sd = np.nanmean(diff), np.nanstd(diff, ddof=1)
        ax.scatter(mean_val, diff, s=16, alpha=0.5, c=DIM_COLORS[dim],
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.axhline(md, color="k", linewidth=0.8, label=f"Mean={md:.3f}")
        ax.axhline(md + 1.96 * sd, color="red", linewidth=0.7, linestyle="--",
                   label=f"+1.96SD={md+1.96*sd:.3f}")
        ax.axhline(md - 1.96 * sd, color="red", linewidth=0.7, linestyle="--",
                   label=f"−1.96SD={md-1.96*sd:.3f}")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.set_xlabel("Mean of Human & VLM"); ax.set_ylabel("Difference (VLM − Human)")
        ax.set_title(DIM_LABELS[dim])
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_xlim(-0.05, 1.05)
    fig.suptitle("Bland–Altman Agreement Plots", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ── Fig 3: Confusion matrices ────────────────────────────────────────────────

def _snap_to_levels(arr: np.ndarray) -> np.ndarray:
    return np.array([LEVELS[np.argmin(np.abs(LEVELS - x))] for x in arr])


def plot_confusion_matrices(data: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    labels = [f"{x:.1f}" for x in LEVELS]

    for ax, dim in zip(axes, DIMS):
        h_snap = _snap_to_levels(data[dim]["human"])
        v_snap = _snap_to_levels(data[dim]["vlm"])

        n = len(LEVELS)
        conf = np.zeros((n, n), dtype=int)
        idx_map = {v: i for i, v in enumerate(LEVELS)}
        for hv, vv in zip(h_snap, v_snap):
            conf[idx_map[hv]][idx_map[vv]] += 1

        im = ax.imshow(conf, cmap="Blues", aspect="equal",
                       vmin=0, vmax=max(conf.max(), 1))
        for i in range(n):
            for j in range(n):
                val = conf[i][j]
                if val > 0:
                    color = "white" if val > conf.max() * 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("VLM Score"); ax.set_ylabel("Human Mean Score")
        ax.set_title(DIM_LABELS[dim])

    fig.suptitle("Score Agreement Confusion Matrices", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig3_confusion_matrix.pdf"))
    _save(_recreate_confusion(data),
          os.path.join(out_dir, "fig3_confusion_matrix.png"))


def _recreate_confusion(data):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    labels = [f"{x:.1f}" for x in LEVELS]
    for ax, dim in zip(axes, DIMS):
        h_snap = _snap_to_levels(data[dim]["human"])
        v_snap = _snap_to_levels(data[dim]["vlm"])
        n = len(LEVELS)
        conf = np.zeros((n, n), dtype=int)
        idx_map = {v: i for i, v in enumerate(LEVELS)}
        for hv, vv in zip(h_snap, v_snap):
            conf[idx_map[hv]][idx_map[vv]] += 1
        im = ax.imshow(conf, cmap="Blues", aspect="equal", vmin=0, vmax=max(conf.max(), 1))
        for i in range(n):
            for j in range(n):
                val = conf[i][j]
                if val > 0:
                    color = "white" if val > conf.max() * 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("VLM Score"); ax.set_ylabel("Human Mean Score")
        ax.set_title(DIM_LABELS[dim])
    fig.suptitle("Score Agreement Confusion Matrices", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ── Fig 4: Radar chart ───────────────────────────────────────────────────────

def plot_radar(metrics: Dict, out_dir: str):
    metric_names = ["Spearman ρ", "Weighted κ", "ICC(2,1)"]
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})

    for dim in DIMS:
        d = metrics["dimensions"][dim]
        values = [
            max(d["spearman_r"], 0),
            max(d["weighted_kappa"], 0),
            max(d["icc21_human_vs_vlm"], 0),
        ]
        values += values[:1]
        ax.plot(angles, values, "-o", linewidth=1.5, markersize=5,
                color=DIM_COLORS[dim], label=DIM_LABELS[dim])
        ax.fill(angles, values, alpha=0.08, color=DIM_COLORS[dim])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Agreement Metrics by Dimension", pad=20, fontsize=10)

    _save(fig, os.path.join(out_dir, "fig4_radar_metrics.pdf"))
    _save(_recreate_radar(metrics), os.path.join(out_dir, "fig4_radar_metrics.png"))


def _recreate_radar(metrics):
    metric_names = ["Spearman ρ", "Weighted κ", "ICC(2,1)"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    for dim in DIMS:
        d = metrics["dimensions"][dim]
        values = [max(d["spearman_r"], 0), max(d["weighted_kappa"], 0),
                  max(d["icc21_human_vs_vlm"], 0)]
        values += values[:1]
        ax.plot(angles, values, "-o", linewidth=1.5, markersize=5,
                color=DIM_COLORS[dim], label=DIM_LABELS[dim])
        ax.fill(angles, values, alpha=0.08, color=DIM_COLORS[dim])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Agreement Metrics by Dimension", pad=20, fontsize=10)
    return fig


# ── Fig 5: Score distribution violin ─────────────────────────────────────────

def plot_score_distribution(data: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, dim in zip(axes, DIMS):
        h = data[dim]["human"]
        v = data[dim]["vlm"]
        color = DIM_COLORS[dim]

        parts_h = ax.violinplot([h[~np.isnan(h)]], positions=[0],
                                showmeans=True, showmedians=True)
        parts_v = ax.violinplot([v[~np.isnan(v)]], positions=[1],
                                showmeans=True, showmedians=True)

        for pc in parts_h["bodies"]:
            pc.set_facecolor("#aaaaaa"); pc.set_alpha(0.6)
        for pc in parts_v["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.6)

        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts_h:
                parts_h[key].set_color("#555555")
            if key in parts_v:
                parts_v[key].set_color(color)

        np.random.seed(42)
        jitter_h = np.random.normal(0, 0.03, size=len(h))
        jitter_v = np.random.normal(0, 0.03, size=len(v))
        ax.scatter(jitter_h, h, s=6, alpha=0.3, c="#555555", zorder=4)
        ax.scatter(1 + jitter_v, v, s=6, alpha=0.3, c=color, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human", "VLM"], fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(DIM_LABELS[dim])

    fig.suptitle("Score Distribution: Human vs. VLM", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig5_distribution_violin.pdf"))
    _save(_recreate_violin(data),
          os.path.join(out_dir, "fig5_distribution_violin.png"))


def _recreate_violin(data):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, dim in zip(axes, DIMS):
        h, v = data[dim]["human"], data[dim]["vlm"]
        color = DIM_COLORS[dim]
        parts_h = ax.violinplot([h[~np.isnan(h)]], positions=[0], showmeans=True, showmedians=True)
        parts_v = ax.violinplot([v[~np.isnan(v)]], positions=[1], showmeans=True, showmedians=True)
        for pc in parts_h["bodies"]: pc.set_facecolor("#aaaaaa"); pc.set_alpha(0.6)
        for pc in parts_v["bodies"]: pc.set_facecolor(color); pc.set_alpha(0.6)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts_h: parts_h[key].set_color("#555555")
            if key in parts_v: parts_v[key].set_color(color)
        np.random.seed(42)
        ax.scatter(np.random.normal(0, 0.03, len(h)), h, s=6, alpha=0.3, c="#555555", zorder=4)
        ax.scatter(1 + np.random.normal(0, 0.03, len(v)), v, s=6, alpha=0.3, c=color, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Human", "VLM"], fontsize=8)
        ax.set_ylabel("Score"); ax.set_ylim(-0.05, 1.05); ax.set_title(DIM_LABELS[dim])
    fig.suptitle("Score Distribution: Human vs. VLM", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ── Fig 6: Per-video deviation bar ───────────────────────────────────────────

def plot_per_video_deviation(merged: pd.DataFrame, data: Dict, out_dir: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    video_ids = merged["video_index"].values
    for ax, dim in zip(axes, DIMS):
        diff = data[dim]["vlm"] - data[dim]["human"]
        colors = [DIM_COLORS[dim] if d >= 0 else "#999999" for d in diff]
        ax.bar(video_ids, diff, color=colors, width=0.8, alpha=0.7, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"Δ {DIM_LABELS[dim]}")
        ax.set_ylim(-1, 1)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.4))

    axes[-1].set_xlabel("Video Index")
    fig.suptitle("Per-Video Score Deviation (VLM − Human Mean)", fontsize=11, y=1.0)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig6_per_video_deviation.pdf"))
    _save(_recreate_deviation(merged, data),
          os.path.join(out_dir, "fig6_per_video_deviation.png"))


def _recreate_deviation(merged, data):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    video_ids = merged["video_index"].values
    for ax, dim in zip(axes, DIMS):
        diff = data[dim]["vlm"] - data[dim]["human"]
        colors = [DIM_COLORS[dim] if d >= 0 else "#999999" for d in diff]
        ax.bar(video_ids, diff, color=colors, width=0.8, alpha=0.7, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"Δ {DIM_LABELS[dim]}"); ax.set_ylim(-1, 1)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.4))
    axes[-1].set_xlabel("Video Index")
    fig.suptitle("Per-Video Score Deviation (VLM − Human Mean)", fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


# ── Fig 7: Summary table ─────────────────────────────────────────────────────

def plot_summary_table(metrics: Dict, out_dir: str):
    cols = ["Dimension", "N", "Human μ", "VLM μ",
            "Spearman ρ", "p-value", "Weighted κ",
            "ICC(2,1)\nH-M", "ICC(2,1)\nInter-H"]
    rows = []
    for dim in DIMS:
        d = metrics["dimensions"][dim]
        rows.append([
            DIM_LABELS[dim],
            d["n_valid"],
            f"{d['human_avg_mean']:.3f}",
            f"{d['vlm_mean']:.3f}",
            f"{d['spearman_r']:.3f}",
            f"{d['spearman_p']:.4f}",
            f"{d['weighted_kappa']:.3f}",
            f"{d['icc21_human_vs_vlm']:.3f}",
            f"{d['icc21_inter_human']:.3f}" if d['icc21_inter_human'] else "—",
        ])

    ov = metrics["overall"]
    rows.append([
        "Overall", ov["n_valid"], "—", "—",
        f"{ov['spearman_r']:.3f}",
        f"{ov['spearman_p']:.4f}",
        "—",
        f"{ov['icc21_final_score']:.3f}",
        "—",
    ])

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        elif r == len(rows):
            cell.set_facecolor("#D6E4F0")
        else:
            cell.set_facecolor("#F2F2F2" if r % 2 == 0 else "white")

    fig.suptitle("Table 1. Human–Machine Consistency Metrics", fontsize=10,
                 y=0.95, fontweight="bold")
    _save(fig, os.path.join(out_dir, "fig7_summary_table.pdf"))
    _save(_recreate_table(metrics), os.path.join(out_dir, "fig7_summary_table.png"))


def _recreate_table(metrics):
    cols = ["Dimension", "N", "Human μ", "VLM μ",
            "Spearman ρ", "p-value", "Weighted κ",
            "ICC(2,1)\nH-M", "ICC(2,1)\nInter-H"]
    rows = []
    for dim in DIMS:
        d = metrics["dimensions"][dim]
        rows.append([DIM_LABELS[dim], d["n_valid"],
            f"{d['human_avg_mean']:.3f}", f"{d['vlm_mean']:.3f}",
            f"{d['spearman_r']:.3f}", f"{d['spearman_p']:.4f}",
            f"{d['weighted_kappa']:.3f}", f"{d['icc21_human_vs_vlm']:.3f}",
            f"{d['icc21_inter_human']:.3f}" if d['icc21_inter_human'] else "—"])
    ov = metrics["overall"]
    rows.append(["Overall", ov["n_valid"], "—", "—",
        f"{ov['spearman_r']:.3f}", f"{ov['spearman_p']:.4f}", "—",
        f"{ov['icc21_final_score']:.3f}", "—"])
    fig, ax = plt.subplots(figsize=(10, 2.2)); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        elif r == len(rows):
            cell.set_facecolor("#D6E4F0")
        else:
            cell.set_facecolor("#F2F2F2" if r % 2 == 0 else "white")
    fig.suptitle("Table 1. Human–Machine Consistency Metrics", fontsize=10,
                 y=0.95, fontweight="bold")
    return fig


# ── Public entry point ────────────────────────────────────────────────────────

def generate_all_figures(
    dataset_json: str,
    annotation_xlsx: str,
    metrics_json: str,
    out_dir: str,
):
    _apply_style()
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    merged, data = _prepare_data(dataset_json, annotation_xlsx)

    with open(metrics_json, encoding="utf-8") as f:
        metrics = json.load(f)

    print("Generating figures...")
    plot_scatter_regression(data, out_dir)
    plot_bland_altman(data, out_dir)
    plot_confusion_matrices(data, out_dir)
    plot_radar(metrics, out_dir)
    plot_score_distribution(data, out_dir)
    plot_per_video_deviation(merged, data, out_dir)
    plot_summary_table(metrics, out_dir)

    print(f"\nAll {7} figures saved to {out_dir}/")
