"""
Part 2.3: 3D trajectories in an 800 x 600 x 400 volume, STL + optimization + Rerun visualization.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from stljax.formula import Always, GreaterThan, Predicate

from .helper import hotspot_times
from .stl_specs import STLThresholds, loss_neg_robustness

VOL_W, VOL_H, VOL_D = 800.0, 600.0, 400.0


def _smooth_norm(v: jnp.ndarray, axis: int = -1, eps: float = 1e-2) -> jnp.ndarray:
    s = jnp.sum(v * v, axis=axis)
    return jnp.sqrt(s + eps * eps)


@dataclass
class Hotspot3D:
    x: float
    y: float
    z: float
    kind: str
    group: list[int]
    time_step: int


@dataclass
class TrajectoryMeta3D:
    starts: np.ndarray
    ends: np.ndarray
    conv: np.ndarray
    div: np.ndarray
    t_conv: int
    t_div: int


def generate_dataset_3d(
    n: int,
    K: int = 60,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, list[Hotspot3D], TrajectoryMeta3D]:
    """
    Returns
    -------
    Ps : (N, K, 3)
    hotspots : list[Hotspot3D]
    meta : TrajectoryMeta3D
    """
    rng = np.random.default_rng(seed)
    t_conv, t_div = hotspot_times(K)
    conv = np.array([250.0, 300.0, 200.0])
    div = np.array([550.0, 300.0, 200.0])
    group = list(range(n))

    starts = np.stack(
        [
            np.array(
                [
                    rng.uniform(5.0, 80.0),
                    rng.uniform(80.0, VOL_H - 80.0),
                    rng.uniform(80.0, VOL_D - 80.0),
                ]
            )
            for _ in range(n)
        ],
        axis=0,
    )
    ends = np.stack(
        [
            np.array(
                [
                    rng.uniform(VOL_W - 80.0, VOL_W - 5.0),
                    rng.uniform(80.0, VOL_H - 80.0),
                    rng.uniform(80.0, VOL_D - 80.0),
                ]
            )
            for _ in range(n)
        ],
        axis=0,
    )

    Ps = np.zeros((n, K, 3), dtype=np.float64)
    for i in range(n):
        for t in range(K):
            if t <= t_conv and t_conv > 0:
                a = t / t_conv
                p = (1.0 - a) * starts[i] + a * conv
            elif t <= t_div:
                a = (t - t_conv) / max(1, (t_div - t_conv))
                mid = (1.0 - a) * conv + a * div
                orth = np.cross(div - conv, np.array([0.0, 0.0, 1.0]))
                orth = orth / (np.linalg.norm(orth) + 1e-6)
                p = mid + orth * rng.normal(0.0, 22.0)
            else:
                a = (t - t_div) / max(1, (K - 1 - t_div))
                p = (1.0 - a) * div + a * ends[i]
            Ps[i, t] = p

    hs = [
        Hotspot3D(float(conv[0]), float(conv[1]), float(conv[2]), "converge", group, t_conv),
        Hotspot3D(float(div[0]), float(div[1]), float(div[2]), "diverge", group, t_div),
    ]
    meta = TrajectoryMeta3D(starts=starts, ends=ends, conv=conv, div=div, t_conv=t_conv, t_div=t_div)
    return Ps, hs, meta


def pos_nkj_to_knj_3d(Ps: np.ndarray) -> np.ndarray:
    return np.transpose(Ps, (1, 0, 2))


def _min_pairwise_dist(pos: jnp.ndarray) -> jnp.ndarray:
    K, N, _ = pos.shape
    if N < 2:
        return jnp.full((K, 1), 1e6, dtype=pos.dtype)
    diffs = pos[:, :, None, :] - pos[:, None, :, :]
    d = _smooth_norm(diffs, axis=-1)
    mask = jnp.triu(jnp.ones((N, N), dtype=d.dtype), k=1)
    big = jnp.array(1e9, dtype=d.dtype)
    d_masked = jnp.where(mask > 0, d, big)
    return jnp.min(d_masked, axis=(1, 2), keepdims=True)


def _bundling_margin(pos: jnp.ndarray, group: list[int], delta_bundle: float) -> jnp.ndarray:
    K, _, _ = pos.shape
    if len(group) < 2:
        return jnp.full((K, 1), 1e6, dtype=pos.dtype)
    idx = jnp.array(group, dtype=jnp.int32)
    pg = pos[:, idx, :]
    G = len(group)
    diffs = pg[:, :, None, :] - pg[:, None, :, :]
    d = _smooth_norm(diffs, axis=-1)
    mask = jnp.triu(jnp.ones((G, G), dtype=d.dtype), k=1)
    margins = jnp.where(mask > 0, delta_bundle - d, jnp.array(1e9, dtype=d.dtype))
    return jnp.min(margins, axis=(1, 2))[:, None]


def _smoothness_margin(pos: jnp.ndarray, a_max: float) -> jnp.ndarray:
    K = pos.shape[0]
    if K < 3:
        return jnp.full((K, 1), 1e6, dtype=pos.dtype)
    acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    norms = _smooth_norm(acc, axis=-1)
    interior = jnp.min(a_max - norms, axis=-1)
    out = jnp.full((K, 1), 1e6, dtype=pos.dtype)
    return out.at[1 : K - 1, 0].set(interior)


def _always_interval(K: int, a: int, b: int) -> list[int]:
    return [max(0, a), min(b, K - 1)]


def build_formulas_3d(meta: TrajectoryMeta3D, thr: STLThresholds, group: list[int], K: int):
    tc, td = meta.t_conv, meta.t_div
    starts = jnp.array(meta.starts)
    ends = jnp.array(meta.ends)
    conv = jnp.array(meta.conv)
    div = jnp.array(meta.div)

    def sep_pred(sig):
        return _min_pairwise_dist(sig) - thr.delta_sep

    def bundle_pred(sig):
        return _bundling_margin(sig, group, thr.delta_bundle)

    def smooth_pred(sig):
        return _smoothness_margin(sig, thr.a_max)

    def start_pred(sig):
        d = _smooth_norm(sig - starts, axis=-1)
        return jnp.min(thr.eps_start - d, axis=-1, keepdims=True)

    def end_pred(sig):
        d = _smooth_norm(sig - ends, axis=-1)
        return jnp.min(thr.eps_end - d, axis=-1, keepdims=True)

    def conv_pred(sig):
        d = _smooth_norm(sig - conv, axis=-1)
        return jnp.min(thr.eps_conv - d, axis=-1, keepdims=True)

    def div_pred(sig):
        d = _smooth_norm(sig - div, axis=-1)
        return jnp.min(thr.eps_div - d, axis=-1, keepdims=True)

    def div_sep_pred(sig):
        return _min_pairwise_dist(sig) - thr.delta_div_sep

    p_sep = Predicate("separation", sep_pred)
    p_bun = Predicate("bundling", bundle_pred)
    p_smo = Predicate("smoothness", smooth_pred)
    p_sta = Predicate("start", start_pred)
    p_end = Predicate("end", end_pred)
    p_con = Predicate("converge_wp", conv_pred)
    p_div = Predicate("diverge_wp", div_pred)
    p_dvs = Predicate("diverge_sep", div_sep_pred)

    f_sep = Always(GreaterThan(p_sep, 0.0), interval=_always_interval(K, 0, K - 1))
    f_bun = Always(GreaterThan(p_bun, 0.0), interval=_always_interval(K, tc, td))
    smo_iv = _always_interval(K, 1, K - 2) if K >= 3 else [0, 0]
    f_smo = Always(GreaterThan(p_smo, 0.0), interval=smo_iv)
    f_sta = Always(GreaterThan(p_sta, 0.0), interval=[0, 0])
    f_con = Always(GreaterThan(p_con, 0.0), interval=[tc, tc])
    f_div = Always(GreaterThan(p_div, 0.0), interval=[td, td])
    f_end = Always(GreaterThan(p_end, 0.0), interval=[K - 1, K - 1])
    f_dvs = Always(GreaterThan(p_dvs, 0.0), interval=[td, td])

    names = ["start", "converge", "bundling", "diverge_wp", "diverge_sep", "end", "separation", "smoothness"]
    formulas = {
        "start": f_sta,
        "converge": f_con,
        "bundling": f_bun,
        "diverge_wp": f_div,
        "diverge_sep": f_dvs,
        "end": f_end,
        "separation": f_sep,
        "smoothness": f_smo,
    }
    combined = f_sta & f_con & f_bun & f_div & f_dvs & f_end & f_sep & f_smo
    return formulas, combined, names


def robustness_specs_3d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta3D,
    thr: STLThresholds | None = None,
    *,
    group: list[int] | None = None,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> dict[str, jnp.ndarray]:
    thr = thr or STLThresholds()
    K = int(pos_knj.shape[0])
    g = group if group is not None else list(range(pos_knj.shape[1]))
    formulas, combined, names = build_formulas_3d(meta, thr, g, K)
    kw = {"approx_method": approx_method, "temperature": temperature}
    out = {n: formulas[n].robustness(pos_knj, **kw) for n in names}
    out["combined"] = combined.robustness(pos_knj, **kw)
    return out


def bundled_losses_3d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta3D,
    thr: STLThresholds,
    group: list[int],
    *,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> dict[str, jnp.ndarray]:
    robs = robustness_specs_3d(pos_knj, meta, thr, group=group, approx_method=approx_method, temperature=temperature)
    position_robs = jnp.stack([robs[k] for k in ["start", "converge", "diverge_wp", "end"]])
    return {
        "bundling_loss": loss_neg_robustness(robs["bundling"]),
        "separation_loss": loss_neg_robustness(robs["separation"]),
        "smoothness_loss": loss_neg_robustness(robs["smoothness"]),
        "position_loss": jnp.mean(loss_neg_robustness(position_robs)),
    }


def total_loss_3d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta3D,
    thr: STLThresholds,
    group: list[int],
    weights: dict[str, float],
    *,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ls = bundled_losses_3d(pos_knj, meta, thr, group, approx_method=approx_method, temperature=temperature)
    w = weights
    total = (
        w.get("bundling", 1.0) * ls["bundling_loss"]
        + w.get("separation", 1.0) * ls["separation_loss"]
        + w.get("smoothness", 1.0) * ls["smoothness_loss"]
        + w.get("position", 1.0) * ls["position_loss"]
    )
    return total, ls


def surrogate_losses_3d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta3D,
    thr: STLThresholds,
    group: list[int],
) -> dict[str, jnp.ndarray]:
    """Same hinge semantics as ``surrogate_losses_2d`` but in $\\mathbb{R}^3$ (Part 2.3 optimization)."""
    K = pos_knj.shape[0]
    tc, td = meta.t_conv, meta.t_div
    starts = jnp.asarray(meta.starts)
    ends = jnp.asarray(meta.ends)
    conv = jnp.asarray(meta.conv)
    div = jnp.asarray(meta.div)

    dmin = _min_pairwise_dist(pos_knj)
    l_sep = jnp.mean(jnp.maximum(0.0, thr.delta_sep - dmin))

    bun = _bundling_margin(pos_knj, group, thr.delta_bundle)
    l_bun = jnp.mean(jnp.maximum(0.0, -bun[tc : td + 1]))

    sm = _smoothness_margin(pos_knj, thr.a_max)
    if K >= 3:
        l_smo = jnp.mean(jnp.maximum(0.0, -sm[1 : K - 1]))
    else:
        l_smo = jnp.array(0.0, dtype=pos_knj.dtype)

    d0 = _smooth_norm(pos_knj[0] - starts, axis=-1)
    l_start = jnp.mean(jnp.maximum(0.0, d0 - thr.eps_start))
    dc = _smooth_norm(pos_knj[tc] - conv, axis=-1)
    l_conv = jnp.mean(jnp.maximum(0.0, dc - thr.eps_conv))
    dd = _smooth_norm(pos_knj[td] - div, axis=-1)
    l_div = jnp.mean(jnp.maximum(0.0, dd - thr.eps_div))
    d1 = _smooth_norm(pos_knj[K - 1] - ends, axis=-1)
    l_end = jnp.mean(jnp.maximum(0.0, d1 - thr.eps_end))
    dmin_div = _min_pairwise_dist(pos_knj[td : td + 1])
    l_div_sep = jnp.mean(jnp.maximum(0.0, thr.delta_div_sep - dmin_div))

    l_pos = (l_start + l_conv + l_div + l_end + 0.5 * l_div_sep) / 4.5

    return {
        "bundling_loss": l_bun,
        "separation_loss": l_sep,
        "smoothness_loss": l_smo,
        "position_loss": l_pos,
    }


def surrogate_total_loss_3d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta3D,
    thr: STLThresholds,
    group: list[int],
    weights: dict[str, float],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ls = surrogate_losses_3d(pos_knj, meta, thr, group)
    w = weights
    total = (
        w.get("bundling", 1.0) * ls["bundling_loss"]
        + w.get("separation", 1.0) * ls["separation_loss"]
        + w.get("smoothness", 1.0) * ls["smoothness_loss"]
        + w.get("position", 1.0) * ls["position_loss"]
    )
    return total, ls


def optimize_trajectories_3d(
    Ps_nkj: np.ndarray,
    meta: TrajectoryMeta3D,
    *,
    steps: int = 500,
    lr: float = 0.02,
    weights: dict[str, float] | None = None,
    thr: STLThresholds | None = None,
) -> tuple[np.ndarray, list[float]]:
    thr = thr or STLThresholds(
        eps_start=40.0,
        eps_conv=50.0,
        eps_div=50.0,
        eps_end=40.0,
        delta_sep=12.0,
        delta_bundle=80.0,
        a_max=180.0,
    )
    weights = weights or {"bundling": 1.0, "separation": 1.0, "smoothness": 0.4, "position": 1.5}
    N = Ps_nkj.shape[0]
    group = list(range(N))
    pos0 = jnp.array(pos_nkj_to_knj_3d(Ps_nkj))

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        t, _ = surrogate_total_loss_3d(p, meta, thr, group, weights)
        return t

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pos0)
    pos = pos0
    hist: list[float] = []

    for _ in range(steps):
        loss, g = value_and_grad(pos)
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        fv = float(loss)
        hist.append(fv)
        if not math.isfinite(fv):
            break
        upd, opt_state = optimizer.update(g, opt_state)
        pos = optax.apply_updates(pos, upd)
        pos = jnp.stack(
            [
                jnp.clip(pos[:, :, 0], 0.0, VOL_W),
                jnp.clip(pos[:, :, 1], 0.0, VOL_H),
                jnp.clip(pos[:, :, 2], 0.0, VOL_D),
            ],
            axis=-1,
        )

    return np.transpose(np.array(pos), (1, 0, 2)), hist


def _pairwise_distances(Ps_nkj: np.ndarray) -> np.ndarray:
    N, K, _ = Ps_nkj.shape
    if N < 2:
        return np.full((K, 1), np.inf)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append(np.linalg.norm(Ps_nkj[i] - Ps_nkj[j], axis=-1))
    return np.stack(pairs, axis=1)


def summarize_metrics_3d(Ps_nkj: np.ndarray, meta: TrajectoryMeta3D, thr: STLThresholds) -> dict[str, float]:
    pos_knj = jnp.asarray(pos_nkj_to_knj_3d(Ps_nkj))
    robs = robustness_specs_3d(pos_knj, meta, thr, group=list(range(Ps_nkj.shape[0])))
    pairwise = _pairwise_distances(Ps_nkj)
    acc = Ps_nkj[:, 2:] - 2 * Ps_nkj[:, 1:-1] + Ps_nkj[:, :-2]
    endpoint_error = np.mean(
        np.r_[
            np.linalg.norm(Ps_nkj[:, 0] - meta.starts, axis=-1),
            np.linalg.norm(Ps_nkj[:, -1] - meta.ends, axis=-1),
        ]
    )
    return {
        "robustness_combined": float(robs["combined"]),
        "robustness_bundling": float(robs["bundling"]),
        "robustness_separation": float(robs["separation"]),
        "robustness_smoothness": float(robs["smoothness"]),
        "robustness_position_min": float(jnp.min(jnp.stack([robs[k] for k in ("start", "converge", "diverge_wp", "end")]))),
        "min_pairwise_distance": float(np.min(pairwise)),
        "max_bundle_pairwise_distance": float(np.max(pairwise[meta.t_conv : meta.t_div + 1])),
        "max_acceleration": float(np.max(np.linalg.norm(acc, axis=-1))) if Ps_nkj.shape[1] >= 3 else 0.0,
        "mean_start_end_error": float(endpoint_error),
    }


def plot_3d_comparison(
    Ps_baseline: np.ndarray,
    Ps_opt: np.ndarray,
    hotspots: list[Hotspot3D],
    out_path: Path,
    *,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for Ps, style, label in ((Ps_baseline, "--", "baseline"), (Ps_opt, "-", "optimized")):
        for i in range(Ps.shape[0]):
            ax.plot(Ps[i, :, 0], Ps[i, :, 1], Ps[i, :, 2], style, alpha=0.65, label=label if i == 0 else None)
    for h in hotspots:
        color = "red" if h.kind == "converge" else "blue"
        marker = "o" if h.kind == "converge" else "s"
        ax.scatter([h.x], [h.y], [h.z], c=color, s=90, marker=marker, label=h.kind)
    ax.set_xlim(0, VOL_W)
    ax.set_ylim(0, VOL_H)
    ax.set_zlim(0, VOL_D)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_rerun(
    Ps_baseline: np.ndarray,
    Ps_opt: np.ndarray,
    hotspots: list[Hotspot3D],
    *,
    app_id: str = "traj3d",
) -> None:
    import rerun as rr

    rr.init(app_id, spawn=True)
    n, k, _ = Ps_baseline.shape
    colors = np.array(
        [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180]],
        dtype=np.uint8,
    )
    for t in range(k):
        rr.set_time_sequence("step", t)
        for i in range(n):
            c = colors[i % len(colors)]
            rr.log(
                f"baseline/agent_{i}",
                rr.Points3D([Ps_baseline[i, t]], colors=[c], radii=[8.0]),
            )
            rr.log(
                f"optimized/agent_{i}",
                rr.Points3D([Ps_opt[i, t]], colors=[c], radii=[8.0]),
            )
        for h in hotspots:
            col = (255, 100, 100) if h.kind == "converge" else (100, 150, 255)
            rr.log(
                f"hotspots/{h.kind}",
                rr.Points3D([[h.x, h.y, h.z]], colors=[col], radii=[25.0]),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-rerun", action="store_true")
    args = ap.parse_args()

    thr = STLThresholds(
        eps_start=40.0,
        eps_conv=50.0,
        eps_div=50.0,
        eps_end=40.0,
        delta_sep=12.0,
        delta_bundle=80.0,
        a_max=180.0,
    )
    Ps, hs, meta = generate_dataset_3d(args.num, K=60, seed=1)
    Ps_opt, hist = optimize_trajectories_3d(Ps, meta, steps=args.steps, thr=thr)

    out = Path("problem2/out3d")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trajectories_3d.npz", baseline=Ps, optimized=Ps_opt, hist=np.array(hist))
    plot_3d_comparison(Ps, Ps_opt, hs, out / "compare_view1.png", elev=25, azim=-65)
    plot_3d_comparison(Ps, Ps_opt, hs, out / "compare_view2.png", elev=15, azim=25)
    plt.figure(figsize=(8, 4))
    plt.plot(hist)
    plt.xlabel("step")
    plt.ylabel("total loss")
    plt.title("3D optimization loss")
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png", dpi=150)
    plt.close()
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": summarize_metrics_3d(Ps, meta, thr),
                "optimized": summarize_metrics_3d(Ps_opt, meta, thr),
                "loss_first": hist[0],
                "loss_last": hist[-1],
            },
            f,
            indent=2,
        )

    if not args.no_rerun:
        try:
            log_rerun(Ps, Ps_opt, hs)
        except Exception as e:
            print("Rerun skipped:", e)

    print("3D optimization done. Loss first/last:", hist[0], hist[-1])
    print("Saved", out / "trajectories_3d.npz")


if __name__ == "__main__":
    main()
