"""
Part 2.2: Adam optimization for 2D trajectories.

Uses hinge surrogates from ``stl_specs.surrogate_total_loss_2d`` (same semantics as the
STL predicates) for stable gradients; report STL robustness via ``robustness_specs`` / ``python -m problem2.stl_specs``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from .helper import export_animation_json, generate_trajectories, pos_nkj_to_knj
from .stl_specs import STLThresholds, robustness_specs, surrogate_losses_2d, surrogate_total_loss_2d


def optimize_trajectories(
    Ps_nkj: np.ndarray,
    meta,
    *,
    steps: int = 400,
    lr: float = 0.08,
    weights: dict[str, float] | None = None,
    thr: STLThresholds | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[float], list[dict]]:
    """Return optimized (N,K,2), total loss history, per-step surrogate loss dicts."""
    thr = thr or STLThresholds()
    weights = weights or {
        "bundling": 1.0,
        "separation": 1.0,
        "smoothness": 0.5,
        "position": 2.0,
    }
    N = Ps_nkj.shape[0]
    group = list(range(N))
    pos0 = jnp.array(pos_nkj_to_knj(Ps_nkj))

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        tot, _ = surrogate_total_loss_2d(p, meta, thr, group, weights)
        return tot

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pos0)
    pos = pos0
    hist: list[float] = []
    details: list[dict] = []
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
                jnp.clip(pos[:, :, 0], 0.0, 800.0),
                jnp.clip(pos[:, :, 1], 0.0, 600.0),
            ],
            axis=-1,
        )
        if len(details) < 5 or _ == steps - 1:
            ls = surrogate_losses_2d(pos, meta, thr, group)
            details.append({k: float(v) for k, v in ls.items()})

    out_nkj = np.array(np.transpose(np.array(pos), (1, 0, 2)))
    return out_nkj, hist, details


def plot_comparison(
    Ps_before: np.ndarray,
    Ps_after: np.ndarray,
    meta,
    out_path: str,
    *,
    title: str = "Baseline vs optimized",
) -> None:
    """Static 2D plot: trajectories and hotspot segments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.set_aspect("equal")
    for Ps, style, lab in (
        (Ps_before, "--", "baseline"),
        (Ps_after, "-", "optimized"),
    ):
        N = Ps.shape[0]
        for i in range(N):
            ax.plot(Ps[i, :, 0], Ps[i, :, 1], style, alpha=0.75, label=lab if i == 0 else None)
    ax.scatter([meta.conv[0]], [meta.conv[1]], c="red", s=120, marker="o", zorder=5, label="converge")
    ax.scatter([meta.div[0]], [meta.div[1]], c="blue", s=120, marker="s", zorder=5, label="diverge")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pairwise_distances(Ps_nkj: np.ndarray) -> np.ndarray:
    """Return distances for unordered trajectory pairs, shape (K, pairs)."""
    N, K, _ = Ps_nkj.shape
    if N < 2:
        return np.full((K, 1), np.inf)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append(np.linalg.norm(Ps_nkj[i] - Ps_nkj[j], axis=-1))
    return np.stack(pairs, axis=1)


def summarize_metrics(Ps_nkj: np.ndarray, meta, thr: STLThresholds) -> dict[str, float]:
    """Report-ready quantitative metrics for baseline/optimized trajectories."""
    pos_knj = jnp.asarray(pos_nkj_to_knj(Ps_nkj))
    robs = robustness_specs(pos_knj, meta, thr, group=list(range(Ps_nkj.shape[0])))
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=2, help="number of trajectories")
    p.add_argument("--k", type=int, default=60, help="time steps")
    p.add_argument("--steps", type=int, default=400, help="Adam steps")
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="problem2/out")
    p.add_argument("--wbundle", type=float, default=1.0)
    p.add_argument("--wsep", type=float, default=1.0)
    p.add_argument("--wsmooth", type=float, default=0.5)
    p.add_argument("--wpos", type=float, default=2.0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Ps, hotspots, meta = generate_trajectories(args.num, args.k, seed=args.seed)
    weights = {
        "bundling": args.wbundle,
        "separation": args.wsep,
        "smoothness": args.wsmooth,
        "position": args.wpos,
    }
    thr = STLThresholds()
    Ps_opt, hist, loss_details = optimize_trajectories(
        Ps,
        meta,
        steps=args.steps,
        lr=args.lr,
        weights=weights,
        thr=thr,
        seed=args.seed + 1,
    )

    export_animation_json(Ps, hotspots, str(out / "baseline.json"))
    export_animation_json(Ps_opt, hotspots, str(out / "optimized.json"))
    plot_comparison(Ps, Ps_opt, meta, str(out / "compare.png"), title=f"N={args.num}, K={args.k}")

    plt.figure(figsize=(8, 4))
    plt.plot(hist)
    plt.xlabel("step")
    plt.ylabel("total loss")
    plt.title("Optimization loss")
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png", dpi=150)
    plt.close()

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": summarize_metrics(Ps, meta, thr),
                "optimized": summarize_metrics(Ps_opt, meta, thr),
                "loss_first": hist[0],
                "loss_last": hist[-1],
                "loss_details": loss_details,
            },
            f,
            indent=2,
        )

    print("Wrote:", out / "baseline.json", out / "optimized.json", out / "compare.png", out / "loss_curve.png", out / "metrics.json")


if __name__ == "__main__":
    main()
