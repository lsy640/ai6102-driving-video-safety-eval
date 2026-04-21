"""Triple-strip 360 panoramic driving video preprocessor.

Each frame has vertical layout:
  y in [0, 260)   : real ground-truth panorama (TOP)
  y in [261, 522) : colored structural annotation (MID)
  y in [522, 784) : generative model output (BOT)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image


from .config import CFG as _C

STRIP_TOP = tuple(_C["preprocessing"]["strip_top"])
STRIP_MID = tuple(_C["preprocessing"]["strip_mid"])
STRIP_BOT = tuple(_C["preprocessing"]["strip_bot"])
FRAME_WIDTH = _C["preprocessing"]["frame_width"]
TOTAL_HEIGHT = _C["preprocessing"]["total_height"]
_DIFF_THRESHOLD = _C["preprocessing"]["diff_threshold"]


@dataclass
class FrameData:
    frame_idx: int
    timestamp: float
    real: np.ndarray
    annotation: np.ndarray
    generated: np.ndarray


@dataclass
class VideoData:
    video_id: str
    num_frames: int
    fps: float
    duration: float
    frames: List[FrameData] = field(default_factory=list)


class DrivingVideoProcessor:
    """Read a triple-strip panoramic mp4 and split into its 3 semantic strips."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_id = os.path.basename(video_path)

    def extract_all_frames(self) -> VideoData:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames: List[FrameData] = []
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fd = FrameData(
                frame_idx=idx,
                timestamp=round(idx / fps, 3),
                real=rgb[STRIP_TOP[0]:STRIP_TOP[1], :, :].copy(),
                annotation=rgb[STRIP_MID[0]:STRIP_MID[1], :, :].copy(),
                generated=rgb[STRIP_BOT[0]:STRIP_BOT[1], :, :].copy(),
            )
            frames.append(fd)
            idx += 1
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from {self.video_path}")

        return VideoData(
            video_id=self.video_id,
            num_frames=len(frames),
            fps=fps,
            duration=len(frames) / fps,
            frames=frames,
        )

    @staticmethod
    def compute_pixel_metrics(data: VideoData) -> List[Dict]:
        """Per-frame MAE / diff-area-pct / PSNR between real and generated strips."""
        out = []
        for fd in data.frames:
            real = fd.real.astype(np.float32)
            gen = fd.generated.astype(np.float32)
            h = min(real.shape[0], gen.shape[0])
            real, gen = real[:h], gen[:h]

            diff = np.abs(real - gen)
            mae = float(diff.mean())
            diff_pct = float((diff.mean(axis=2) > _DIFF_THRESHOLD).mean())
            mse = float(((real - gen) ** 2).mean())
            psnr = float(10 * np.log10(255.0 ** 2 / (mse + 1e-8)))

            out.append({
                "frame_idx": fd.frame_idx,
                "timestamp": fd.timestamp,
                "mae": round(mae, 2),
                "diff_area_pct": round(diff_pct * 100, 1),
                "psnr": round(psnr, 2),
            })
        return out

    @staticmethod
    def compute_temporal_metrics(pixel_metrics: List[Dict]) -> Dict:
        maes = [m["mae"] for m in pixel_metrics]
        diffs = [m["diff_area_pct"] for m in pixel_metrics]
        x = np.arange(len(maes))
        slope = float(np.polyfit(x, maes, 1)[0]) if len(maes) >= 2 else 0.0
        deltas = np.diff(maes) if len(maes) >= 2 else np.array([0.0])
        return {
            "mae_slope": round(slope, 3),
            "volatility": round(float(np.std(deltas)), 3),
            "max_frame_jump": round(float(np.max(np.abs(deltas))), 2),
            "final_diff_pct": diffs[-1] if diffs else 0.0,
        }

    @staticmethod
    def parse_annotation_layer(annotation_frame: np.ndarray) -> Dict:
        """Extract coarse scene-structure info from the colored annotation strip."""
        ann = annotation_frame.astype(np.float32)
        r, g, b = ann[:, :, 0], ann[:, :, 1], ann[:, :, 2]

        red = (r > 100) & (g < 80) & (b < 80)
        blue = (b > 100) & (r < 80) & (g < 80)
        green = (g > 100) & (r < 80) & (b < 80)
        yellow = (r > 100) & (g > 100) & (b < 80)

        total = ann.shape[0] * ann.shape[1]
        return {
            "obstacle_density": round(float(red.sum()) / total, 4),
            "lane_line_density": round(float(blue.sum()) / total, 4),
            "crosswalk_density": round(float(green.sum()) / total, 4),
            "signal_density": round(float(yellow.sum()) / total, 4),
            "has_obstacles": bool(red.sum() > 100),
            "has_lane_lines": bool(blue.sum() > 50),
            "has_crosswalk": bool(green.sum() > 50),
            "has_signals": bool(yellow.sum() > 30),
            "scene_complexity": round(
                float(red.sum() + blue.sum() + green.sum()) / total, 4
            ),
        }

    @staticmethod
    def frame_to_pil(arr: np.ndarray, resize=None) -> Image.Image:
        img = Image.fromarray(arr)
        if resize is not None:
            img = img.resize(resize, Image.LANCZOS)
        return img

    @staticmethod
    def stack_real_gen(real: np.ndarray, gen: np.ndarray, resize=(1344, 130)) -> Image.Image:
        """Vertically stack real (top) and generated (bottom) strips for VLM input."""
        real_img = Image.fromarray(real).resize(resize, Image.LANCZOS)
        gen_img = Image.fromarray(gen).resize(resize, Image.LANCZOS)
        w, h = real_img.size
        combo = Image.new("RGB", (w, h * 2 + 4), (0, 0, 0))
        combo.paste(real_img, (0, 0))
        combo.paste(gen_img, (0, h + 4))
        return combo
