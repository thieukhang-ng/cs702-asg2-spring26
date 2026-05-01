"""
Signal Temporal Logic (stljax) specifications for trajectory animation (Part 2.1).

Convention: ``pos`` has shape (K, N, 2) — time, agents, xy. Use ``layout="knj"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from stljax.formula import Always, GreaterThan, Predicate

from .helper import TrajectoryMeta, generate_trajectories, pos_nkj_to_knj


def _smooth_norm(v: jnp.ndarray, axis: int = -1, eps: float = 1e-2) -> jnp.ndarray:
    """Euclidean norm with floor on squared length so $\\nabla\\|x\\|$ is finite at $x=0$ (needed for optimization)."""
    s = jnp.sum(v * v, axis=axis)
    return jnp.sqrt(s + eps * eps)


@dataclass(frozen=True)
class STLThresholds:
    eps_start: float = 25.0
    eps_conv: float = 35.0
    eps_div: float = 35.0
    eps_end: float = 25.0
    delta_sep: float = 8.0
    delta_bundle: float = 45.0
    a_max: float = 120.0
    delta_div_sep: float = 5.0


def _min_pairwise_dist(pos: jnp.ndarray) -> jnp.ndarray:
    """pos (K,N,2) -> (K,1) minimum distance over all unordered pairs."""
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
    """All-pairs bundling margin: min_{i<j in group} (delta_bundle - dist_ij), shape (K,1)."""
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
    """Discrete acceleration a_t = p_{t+1}-2p_t+p_{t-1}; margin min_i (a_max - ||a||), shape (K,1)."""
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


def build_formulas_for_K(meta: TrajectoryMeta, thr: STLThresholds, group: list[int], K: int):
    """Instantiate temporal intervals with concrete K (stljax clips [0, K-1])."""
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
    if K >= 3:
        smo_iv = _always_interval(K, 1, K - 2)
    else:
        smo_iv = [0, 0]
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


def robustness_specs(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta,
    thr: STLThresholds | None = None,
    *,
    group: list[int] | None = None,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> dict[str, jnp.ndarray]:
    """Scalar robustness per named specification + ``combined``."""
    thr = thr or STLThresholds()
    K = int(pos_knj.shape[0])
    g = group if group is not None else list(range(pos_knj.shape[1]))
    formulas, combined, names = build_formulas_for_K(meta, thr, g, K)
    kw = {"approx_method": approx_method, "temperature": temperature}
    out = {n: formulas[n].robustness(pos_knj, **kw) for n in names}
    out["combined"] = combined.robustness(pos_knj, **kw)
    return out


def loss_neg_robustness(r: jnp.ndarray) -> jnp.ndarray:
    """Differentiable penalty: push robustness upward."""
    return jnp.maximum(0.0, -r)


def bundled_losses(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta,
    thr: STLThresholds,
    group: list[int],
    *,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> dict[str, jnp.ndarray]:
    """Per-term loss for Part 2.2 (non-negative)."""
    robs = robustness_specs(pos_knj, meta, thr, group=group, approx_method=approx_method, temperature=temperature)
    position_robs = jnp.stack([robs[k] for k in ["start", "converge", "diverge_wp", "end"]])
    out = {
        "bundling_loss": loss_neg_robustness(robs["bundling"]),
        "separation_loss": loss_neg_robustness(robs["separation"]),
        "smoothness_loss": loss_neg_robustness(robs["smoothness"]),
        "position_loss": jnp.mean(loss_neg_robustness(position_robs)),
    }
    return out


def total_loss_2d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta,
    thr: STLThresholds,
    group: list[int],
    weights: dict[str, float],
    *,
    approx_method: str = "logsumexp",
    temperature: float = 20.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ls = bundled_losses(pos_knj, meta, thr, group, approx_method=approx_method, temperature=temperature)
    w = weights
    total = (
        w.get("bundling", 1.0) * ls["bundling_loss"]
        + w.get("separation", 1.0) * ls["separation_loss"]
        + w.get("smoothness", 1.0) * ls["smoothness_loss"]
        + w.get("position", 1.0) * ls["position_loss"]
    )
    return total, ls


def surrogate_losses_2d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta,
    thr: STLThresholds,
    group: list[int],
) -> dict[str, jnp.ndarray]:
    """
    Differentiable losses aligned with the STL predicates in ``build_formulas_for_K``.
    Used for Part 2.2 optimization: ``stljax`` robustness is used for Part 2.1 *evaluation*,
    but its VJP is often zero for these temporal aggregations, so we optimize these hinges instead.
    """
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


def surrogate_total_loss_2d(
    pos_knj: jnp.ndarray,
    meta: TrajectoryMeta,
    thr: STLThresholds,
    group: list[int],
    weights: dict[str, float],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ls = surrogate_losses_2d(pos_knj, meta, thr, group)
    w = weights
    total = (
        w.get("bundling", 1.0) * ls["bundling_loss"]
        + w.get("separation", 1.0) * ls["separation_loss"]
        + w.get("smoothness", 1.0) * ls["smoothness_loss"]
        + w.get("position", 1.0) * ls["position_loss"]
    )
    return total, ls


def print_robustness_table(Ns: list[int], K: int = 10, *, seed: int = 0) -> None:
    """Part 2.1.2-style table (printed to stdout)."""
    thr = STLThresholds()
    print(f"Robustness (K={K}, logsumexp T=20), columns = specs + combined\n")
    header = ["N", "start", "conv", "bundle", "div", "divSep", "end", "sep", "smooth", "comb"]
    print(" | ".join(f"{h:>8}" for h in header))
    print("-" * (10 * len(header)))
    for N in Ns:
        Ps, _, meta = generate_trajectories(N, K, seed=seed)
        pos = jnp.asarray(pos_nkj_to_knj(Ps))
        r = robustness_specs(pos, meta, thr, group=list(range(N)), approx_method="logsumexp", temperature=20.0)
        row = [
            str(N),
            f"{float(r['start']):.2f}",
            f"{float(r['converge']):.2f}",
            f"{float(r['bundling']):.2f}",
            f"{float(r['diverge_wp']):.2f}",
            f"{float(r['diverge_sep']):.2f}",
            f"{float(r['end']):.2f}",
            f"{float(r['separation']):.2f}",
            f"{float(r['smoothness']):.2f}",
            f"{float(r['combined']):.2f}",
        ]
        print(" | ".join(f"{x:>8}" for x in row))


if __name__ == "__main__":
    print("=== Part 2.1.1: N=2, K=10 ===")
    Ps, _, meta = generate_trajectories(2, 10, seed=0)
    pos = jnp.asarray(pos_nkj_to_knj(Ps))
    thr = STLThresholds()
    r = robustness_specs(pos, meta, thr, group=[0, 1], approx_method="logsumexp", temperature=20.0)
    for k, v in sorted(r.items()):
        print(f"  {k:12} robustness = {float(v):.4f}")
    print("\n=== Part 2.1.2: N in {2,3,5,10} ===")
    print_robustness_table([2, 3, 5, 10], K=10, seed=0)
