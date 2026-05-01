"""
Synthetic 2D trajectories on an 800x600 canvas with convergence / divergence hotspots.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class Hotspot:
    x: float
    y: float
    kind: str  # "converge" | "diverge"
    group: list[int]
    time_step: int


@dataclass
class TrajectoryMeta:
    """Extra data for STL position constraints (not serialized to minimal JSON unless needed)."""

    starts: np.ndarray  # (N, 2)
    ends: np.ndarray  # (N, 2)
    conv: np.ndarray  # (2,)
    div: np.ndarray  # (2,)
    t_conv: int
    t_div: int


CANVAS_W, CANVAS_H = 800.0, 600.0


def hotspot_times(K: int) -> tuple[int, int]:
    """Match assignment scaling: K=10 -> t_conv=2, t_div=7."""
    if K < 3:
        return 0, min(1, K - 1)
    t_conv = min(K - 2, max(1, (K * 2) // 10))
    t_div = min(K - 2, max(t_conv + 1, (K * 7) // 10))
    return t_conv, t_div


def generate_trajectories(
    N: int,
    K: int,
    *,
    seed: int = 0,
    conv_xy: tuple[float, float] | None = None,
    div_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[Hotspot], TrajectoryMeta]:
    """
    Returns
    -------
    Ps : ndarray, shape (N, K, 2)
        Positions in pixel coordinates.
    hotspots : list[Hotspot]
    meta : TrajectoryMeta
        Starts, ends, hotspot locations, and integer times for STL specs.
    """
    rng = np.random.default_rng(seed)
    t_conv, t_div = hotspot_times(K)
    conv = np.array(conv_xy if conv_xy is not None else (250.0, 300.0), dtype=np.float64)
    div = np.array(div_xy if div_xy is not None else (550.0, 300.0), dtype=np.float64)

    starts = np.stack(
        [
            np.array(
                [
                    rng.uniform(30.0, 130.0),
                    rng.uniform(80.0, CANVAS_H - 80.0),
                ]
            )
            for _ in range(N)
        ],
        axis=0,
    )
    ends = np.stack(
        [
            np.array(
                [
                    rng.uniform(CANVAS_W - 130.0, CANVAS_W - 30.0),
                    rng.uniform(80.0, CANVAS_H - 80.0),
                ]
            )
            for _ in range(N)
        ],
        axis=0,
    )

    Ps = np.zeros((N, K, 2), dtype=np.float64)
    group = list(range(N))
    for i in range(N):
        for t in range(K):
            if t <= t_conv and t_conv > 0:
                a = t / t_conv
                p = (1.0 - a) * starts[i] + a * conv
            elif t <= t_div:
                a = (t - t_conv) / max(1, (t_div - t_conv))
                mid = (1.0 - a) * conv + a * div
                # Perpendicular jitter breaks perfect bundling (negative bundling robustness before opt).
                orth = np.array([-(div[1] - conv[1]), div[0] - conv[0]])
                orth = orth / (np.linalg.norm(orth) + 1e-6)
                p = mid + orth * rng.normal(0.0, 18.0)
            else:
                a = (t - t_div) / max(1, (K - 1 - t_div))
                p = (1.0 - a) * div + a * ends[i]
            Ps[i, t] = p

    hotspots = [
        Hotspot(float(conv[0]), float(conv[1]), "converge", group, t_conv),
        Hotspot(float(div[0]), float(div[1]), "diverge", group, t_div),
    ]
    meta = TrajectoryMeta(starts=starts, ends=ends, conv=conv, div=div, t_conv=t_conv, t_div=t_div)
    return Ps, hotspots, meta


def export_animation_json(
    Ps: np.ndarray,
    hotspots: list[Hotspot],
    path: str,
    *,
    colors: list[list[int]] | None = None,
    fps: float = 30.0,
) -> None:
    """Write trajectories and hotspot metadata for ``animate.py``."""
    N = Ps.shape[0]
    if colors is None:
        cmap = [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
        ]
        colors = [cmap[i % len(cmap)] for i in range(N)]

    payload: dict[str, Any] = {
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "fps": fps,
        "trajectories": Ps.tolist(),
        "colors": colors,
        "hotspots": [asdict(h) for h in hotspots],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_animation_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pos_nkj_to_knj(Ps: np.ndarray) -> np.ndarray:
    """(N, K, 2) -> (K, N, 2) for JAX/STL code."""
    return np.transpose(Ps, (1, 0, 2))
