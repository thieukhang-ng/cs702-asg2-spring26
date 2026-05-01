from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def build_state_space(
    M: np.ndarray, B: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build continuous-time state-space matrices for endpoint impedance."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    n = M.shape[0]
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    m_inv = np.linalg.inv(M)

    A_c = np.block([[zeros, eye], [-m_inv @ K, -m_inv @ B]])
    B_c = np.vstack([zeros, m_inv])
    C = np.hstack([eye, zeros])
    D = np.zeros((n, n))
    return A_c, B_c, C, D


def simulate_step_reach(
    M: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    p_start: np.ndarray,
    p_target: np.ndarray,
    T_total: float = 2.0,
    dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a constant-equilibrium step reach with zero external force."""
    p_start = np.asarray(p_start, dtype=float).reshape(-1)
    p_target = np.asarray(p_target, dtype=float).reshape(-1)
    n = p_start.size
    A_c, B_c, C, D = build_state_space(M, B, K)
    system = signal.StateSpace(A_c, B_c, C, D)
    t = np.arange(0.0, T_total + dt / 2.0, dt)
    u = np.zeros_like(t) if n == 1 else np.zeros((t.size, n))
    x0 = np.concatenate([p_start - p_target, np.zeros(n)])
    _, _, states = signal.lsim(system, U=u, T=t, X0=x0)
    states = np.asarray(states)
    if states.ndim == 1:
        states = states.reshape(-1, 2)
    pos = p_target + states[:, :n]
    vel = states[:, n:]
    return t, pos, vel, states


def min_jerk_profile(t: np.ndarray, T: float) -> np.ndarray:
    """Minimum-jerk interpolation profile from 0 to 1 over duration T."""
    tau = np.clip(np.asarray(t, dtype=float) / T, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def simulate_smooth_reaching(
    M: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    p_start: np.ndarray,
    p_target: np.ndarray,
    T_reach: float = 0.5,
    T_total: float = 1.0,
    dt: float = 1 / 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate reaching while the equilibrium follows a minimum-jerk path."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    p_start = np.asarray(p_start, dtype=float).reshape(-1)
    p_target = np.asarray(p_target, dtype=float).reshape(-1)
    n = p_start.size
    m_inv = np.linalg.inv(M)
    direction = p_target - p_start

    def dynamics(t_value: float, state: np.ndarray) -> np.ndarray:
        pos = state[:n]
        vel = state[n:]
        s = float(min_jerk_profile(np.array([t_value]), T_reach)[0])
        p_eq = p_start + s * direction
        acc = m_inv @ (-B @ vel - K @ (pos - p_eq))
        return np.concatenate([vel, acc])

    x0 = np.concatenate([p_start, np.zeros(n)])
    t_eval = np.arange(0.0, T_total + dt / 2.0, dt)
    sol = solve_ivp(
        dynamics,
        (0.0, T_total),
        x0,
        t_eval=t_eval,
        max_step=dt,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Smooth reaching simulation failed: {sol.message}")

    t = sol.t
    pos = sol.y[:n].T
    vel = sol.y[n:].T
    p_eq = p_start + min_jerk_profile(t, T_reach)[:, None] * direction
    return t, pos, vel, p_eq


def settling_time(
    t: np.ndarray, pos: np.ndarray, target: float, amplitude: float, tolerance: float = 0.02
) -> float:
    """Return the first time after which position stays within tolerance band."""
    band = tolerance * abs(amplitude)
    inside = np.abs(pos - target) <= band
    for idx in range(t.size):
        if np.all(inside[idx:]):
            return float(t[idx])
    return float("nan")


def damping_metrics(
    t: np.ndarray, pos: np.ndarray, start: float, target: float, m: float, b: float, k: float
) -> dict[str, float]:
    amplitude = target - start
    direction = np.sign(amplitude) if amplitude != 0 else 1.0
    signed_error = direction * (pos - target)
    overshoot = max(0.0, float(np.max(signed_error))) / abs(amplitude) * 100.0
    return {
        "b": b,
        "zeta": b / (2.0 * np.sqrt(m * k)),
        "peak_position": float(pos[np.argmax(signed_error)]),
        "overshoot_percent": overshoot,
        "settling_time_s": settling_time(t, pos, target, amplitude),
    }


def speed(vel: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vel, axis=1)


def savefig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def format_float(value: float) -> str:
    if np.isnan(value):
        return "--"
    return f"{value:.4f}"


def latex_bmatrix(matrix: np.ndarray, precision: int = 4) -> str:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    rows = []
    for row in matrix:
        cleaned = [0.0 if abs(value) < 1e-10 else value for value in row]
        rows.append(" & ".join(f"{value:.{precision}g}" for value in cleaned))
    return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"


def write_damping_tables(rows: list[dict[str, float]]) -> None:
    csv_path = TABLE_DIR / "damping_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "b",
                "zeta",
                "peak_position",
                "overshoot_percent",
                "settling_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    tex_path = TABLE_DIR / "damping_metrics.tex"
    lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$b$ (N s/m) & $\\zeta$ & Peak $x$ (m) & Overshoot (\\%) & Settling time (s) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['b']:.1f} & {row['zeta']:.3f} & {row['peak_position']:.4f} & "
            f"{row['overshoot_percent']:.2f} & {format_float(row['settling_time_s'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines))


def write_matrix_summary(
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> None:
    parts = []
    for label, (A_c, B_c, C, D) in matrices.items():
        slug = label.lower().replace(" ", "_")
        single = [
            "\\begin{align*}",
            f"A_c &= {latex_bmatrix(A_c)},\\\\",
            f"B_c &= {latex_bmatrix(B_c)},\\\\",
            f"C &= {latex_bmatrix(C)},\\\\",
            f"D &= {latex_bmatrix(D)}.",
            "\\end{align*}",
            "",
        ]
        (TABLE_DIR / f"state_matrices_{slug}.tex").write_text("\n".join(single))

        parts.append(f"\\paragraph{{{label}}}")
        parts.extend(line for line in single if line)
    (TABLE_DIR / "state_matrices.tex").write_text("\n".join(parts) + "\n")


def plot_1d_step(t: np.ndarray, pos: np.ndarray, vel: np.ndarray, target: float) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    axes[0].plot(t, pos[:, 0], color="#1f77b4", label="$x(t)$")
    axes[0].axhline(target, color="#444444", linestyle="--", linewidth=1.0, label="target")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vel[:, 0], color="#d62728", label="$\\dot{x}(t)$")
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("1D step response after equilibrium shift")
    return savefig(fig, "one_d_step_response.png")


def plot_damping_study(results: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]], target: float) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    for b, (t, pos, vel) in results.items():
        axes[0].plot(t, pos[:, 0], label=f"$b={b:g}$")
        axes[1].plot(t, vel[:, 0], label=f"$b={b:g}$")
    axes[0].axhline(target, color="#444444", linestyle="--", linewidth=1.0, label="target")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Effect of damping on 1D reaching")
    return savefig(fig, "one_d_damping_study.png")


def plot_reach_panels(
    t: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    p_start: np.ndarray,
    p_target: np.ndarray,
    title: str,
    filename: str,
    p_eq: np.ndarray | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    axes[0].plot(pos[:, 0], pos[:, 1], color="#1f77b4", label="hand")
    if p_eq is not None:
        axes[0].plot(p_eq[:, 0], p_eq[:, 1], color="#ff7f0e", linestyle="--", label="$p_{eq}$")
    axes[0].scatter(p_start[0], p_start[1], marker="o", color="#2ca02c", label="start")
    axes[0].scatter(p_target[0], p_target[1], marker="x", color="#d62728", label="target")
    axes[0].set_xlabel("$x$ (m)")
    axes[0].set_ylabel("$y$ (m)")
    axes[0].axis("equal")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, pos[:, 0], label="$x(t)$")
    axes[1].plot(t, pos[:, 1], label="$y(t)$")
    if p_eq is not None:
        axes[1].plot(t, p_eq[:, 0], linestyle="--", label="$x_{eq}(t)$")
        axes[1].plot(t, p_eq[:, 1], linestyle="--", label="$y_{eq}(t)$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (m)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, speed(vel), color="#9467bd")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    return savefig(fig, filename)


def plot_duration_study(
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for duration, (t, _pos, vel, _p_eq) in curves.items():
        ax.plot(t, speed(vel), label=f"$T_{{reach}}={duration:g}$ s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Reach-duration study with minimum-jerk equilibrium shift")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return savefig(fig, "reach_duration_speed_profiles.png")


def plot_3d_reach(
    t: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    p_eq: np.ndarray,
    p_start: np.ndarray,
    p_target: np.ndarray,
) -> Path:
    fig = plt.figure(figsize=(12.0, 4.0))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="hand", color="#1f77b4")
    ax3d.plot(p_eq[:, 0], p_eq[:, 1], p_eq[:, 2], linestyle="--", label="$p_{eq}$", color="#ff7f0e")
    ax3d.scatter(*p_start, marker="o", color="#2ca02c", label="start")
    ax3d.scatter(*p_target, marker="x", color="#d62728", label="target")
    ax3d.set_xlabel("$x$ (m)")
    ax3d.set_ylabel("$y$ (m)")
    ax3d.set_zlabel("$z$ (m)")
    ax3d.legend(loc="best")
    ax3d.set_title("3D trajectory")

    ax_pos = fig.add_subplot(1, 3, 2)
    labels = ["$x(t)$", "$y(t)$", "$z(t)$"]
    for idx, label in enumerate(labels):
        ax_pos.plot(t, pos[:, idx], label=label)
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(loc="best")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_title("Position components")

    ax_speed = fig.add_subplot(1, 3, 3)
    ax_speed.plot(t, speed(vel), color="#9467bd")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_title("Speed profile")
    fig.suptitle("3D smooth reaching simulation")
    return savefig(fig, "three_d_smooth_reach.png")


def run() -> None:
    ensure_output_dirs()

    m = 1.5
    b = 15.0
    k = 200.0
    M_1d = np.array([[m]])
    B_1d = np.array([[b]])
    K_1d = np.array([[k]])
    x_start = np.array([0.30])
    x_target = np.array([0.45])

    A_1d, Bc_1d, C_1d, D_1d = build_state_space(M_1d, B_1d, K_1d)
    t_1d, pos_1d, vel_1d, _ = simulate_step_reach(
        M_1d, B_1d, K_1d, x_start, x_target, T_total=2.0
    )
    plot_1d_step(t_1d, pos_1d, vel_1d, x_target[0])

    damping_results = {}
    damping_rows = []
    for b_value in [5.0, 15.0, 40.0]:
        B_current = np.array([[b_value]])
        t_d, pos_d, vel_d, _ = simulate_step_reach(
            M_1d, B_current, K_1d, x_start, x_target, T_total=2.0
        )
        damping_results[b_value] = (t_d, pos_d, vel_d)
        damping_rows.append(
            damping_metrics(t_d, pos_d[:, 0], x_start[0], x_target[0], m, b_value, k)
        )
    plot_damping_study(damping_results, x_target[0])
    write_damping_tables(damping_rows)

    M_2d = np.array([[1.5, 0.3], [0.3, 2.0]])
    B_2d = np.array([[15.0, 3.0], [3.0, 20.0]])
    K_2d = np.array([[200.0, 50.0], [50.0, 350.0]])
    p_start_2d = np.array([0.30, 0.20])
    p_target_2d = np.array([0.45, 0.35])
    A_2d, Bc_2d, C_2d, D_2d = build_state_space(M_2d, B_2d, K_2d)

    t_2d, pos_2d, vel_2d, _ = simulate_step_reach(
        M_2d, B_2d, K_2d, p_start_2d, p_target_2d, T_total=2.0
    )
    plot_reach_panels(
        t_2d,
        pos_2d,
        vel_2d,
        p_start_2d,
        p_target_2d,
        "2D step response after equilibrium shift",
        "two_d_step_response.png",
    )

    t_mj, pos_mj, vel_mj, p_eq_mj = simulate_smooth_reaching(
        M_2d, B_2d, K_2d, p_start_2d, p_target_2d, T_reach=0.5, T_total=1.0
    )
    plot_reach_panels(
        t_mj,
        pos_mj,
        vel_mj,
        p_start_2d,
        p_target_2d,
        "2D minimum-jerk equilibrium shift",
        "two_d_minimum_jerk_response.png",
        p_eq=p_eq_mj,
    )

    duration_curves = {}
    for duration in [0.3, 0.5, 0.8, 1.2]:
        duration_curves[duration] = simulate_smooth_reaching(
            M_2d,
            B_2d,
            K_2d,
            p_start_2d,
            p_target_2d,
            T_reach=duration,
            T_total=duration + 0.5,
        )
    plot_duration_study(duration_curves)

    M_3d = np.array([[1.5, 0.3, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 1.0]])
    B_3d = np.array([[15.0, 4.0, 0.0], [4.0, 25.0, 0.0], [0.0, 0.0, 10.0]])
    K_3d = np.array([[200.0, 60.0, 0.0], [60.0, 400.0, 0.0], [0.0, 0.0, 100.0]])
    p_start_3d = np.array([0.25, 0.20, 0.10])
    p_target_3d = np.array([0.40, 0.40, 0.30])
    A_3d, Bc_3d, C_3d, D_3d = build_state_space(M_3d, B_3d, K_3d)

    t_3d, pos_3d, vel_3d, p_eq_3d = simulate_smooth_reaching(
        M_3d, B_3d, K_3d, p_start_3d, p_target_3d, T_reach=0.5, T_total=1.0
    )
    plot_3d_reach(t_3d, pos_3d, vel_3d, p_eq_3d, p_start_3d, p_target_3d)

    write_matrix_summary(
        {
            "1D system": (A_1d, Bc_1d, C_1d, D_1d),
            "2D system": (A_2d, Bc_2d, C_2d, D_2d),
            "3D system": (A_3d, Bc_3d, C_3d, D_3d),
        }
    )

    print("Generated figures:")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")
    print("Generated tables:")
    for path in sorted(TABLE_DIR.glob("*")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
