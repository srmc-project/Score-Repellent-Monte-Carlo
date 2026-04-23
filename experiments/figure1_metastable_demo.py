#!/usr/bin/env python3
"""Illustrative figure generator for score-based MCMC (ICML intro, 2x4 panel).

This is an *improved* version of `make_score_based_mcmc_intro_figure.py` aimed at
making the *history-dependent change of the vector field* visually obvious.

Key design choices
------------------
Target π:
  A 2D Gaussian mixture with (i) a deep/narrow "trap" at (-2,0) and (ii) a broad
  mode at (+2,0). The energy landscape is U(x) = -log π(x).

Baseline sampler:
  Overdamped Langevin (Euler–Maruyama):
    x_{n+1} = x_n + ε s(x_n) + sqrt(2ε) ξ_n,  s(x)=∇log π(x).

Score-based MCMC sampler (illustrative dynamics):
  Uses an adapted target
    π_{θ}(x) ∝ π(x) exp(-α θ^T s(x))
  which implies an adapted score field
    ∇log π_{θ}(x) = s(x) - α H_{log π}(x) θ,
  where θ is a running (EMA) average of past scores.

Plotting tweaks vs v1:
  • heatmap background (top: U, bottom: U_θ)
  • thicker / longer arrows
  • bottom row overlays baseline arrows (faint gray) + adapted arrows (orange)
  • arrow magnitudes are clipped for readability

Outputs:
  /mnt/data/score_based_mcmc_intro_figure_v3.pdf
  /mnt/data/score_based_mcmc_intro_figure_v3.png

Dependencies: numpy, matplotlib (no SciPy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


# ---------- Target distribution: 2D Gaussian mixture (deep narrow well + broad mode) ----------

def gaussian_density(x: np.ndarray, m: np.ndarray, Sinv: np.ndarray, logdet: float, dim: int) -> np.ndarray:
    """Multivariate Gaussian density N(m, S) evaluated at x."""
    dx = x - m
    quad = np.einsum("...i,ij,...j->...", dx, Sinv, dx)
    log_norm = -0.5 * (dim * np.log(2 * np.pi) + logdet)
    return np.exp(log_norm - 0.5 * quad)


def mixture_logp_grad_hess(x: np.ndarray, comps: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For π(x)=Σ_k w_k N(x; m_k, Σ_k), compute logπ, score, Hessian of logπ."""
    x = np.atleast_2d(x)
    n, dim = x.shape

    phis = []
    grad_ps = []
    hess_ps = []

    for c in comps:
        w, m, Sinv, logdet = c["w"], c["m"], c["Sinv"], c["logdet"]
        phi = gaussian_density(x, m, Sinv, logdet, dim)  # (n,)
        phis.append(w * phi)

        # score of component: ∇ log N(x; m, Σ) = Σ^{-1}(m - x)
        score_k = (m - x) @ Sinv.T  # (n, dim)

        # ∇ (w φ) = (w φ) score_k
        grad_ps.append((w * phi)[:, None] * score_k)

        # ∇^2 (φ) = φ (score score^T - Σ^{-1})
        outer = np.einsum("ni,nj->nij", score_k, score_k)
        hess_phi = phi[:, None, None] * (outer - Sinv)
        hess_ps.append(w * hess_phi)

    p = np.sum(phis, axis=0)  # (n,)
    grad_p = np.sum(grad_ps, axis=0)  # (n, dim)
    hess_p = np.sum(hess_ps, axis=0)  # (n, dim, dim)

    score = grad_p / p[:, None]
    outer_grad = np.einsum("ni,nj->nij", grad_p, grad_p) / (p[:, None, None] ** 2)
    hess_logp = hess_p / p[:, None, None] - outer_grad
    logp = np.log(p)
    return logp, score, hess_logp


# Mixture parameters (one narrow component as a "trap")
Sigma_trap = np.array([[0.18, 0.0], [0.0, 0.18]]) ** 2
Sigma_broad = np.array([[1.0, 0.0], [0.0, 1.0]]) ** 2

comps = [
    dict(w=0.80, m=np.array([-2.0, 0.0]), Sinv=np.linalg.inv(Sigma_trap), logdet=np.log(np.linalg.det(Sigma_trap))),
    dict(w=0.20, m=np.array([+2.0, 0.0]), Sinv=np.linalg.inv(Sigma_broad), logdet=np.log(np.linalg.det(Sigma_broad))),
]


# ---------- Simulators ----------

def simulate_langevin(
    n_steps: int,
    eps: float,
    seed: int,
    adaptive: bool,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler–Maruyama overdamped Langevin with optional score-based adaptation.

    Returns
    -------
    traj : (n_steps+1, 2)
        Includes the initial state at index 0.
    thetas : (n_steps+1, 2)
        Running score average (zeros if adaptive=False). Includes θ_0.
    """
    rng = np.random.default_rng(seed)

    # Start *inside* the deep well to emphasize trapping.
    x = np.array([-2.0, 0.0], dtype=float)
    theta = np.zeros(2, dtype=float)

    traj = np.zeros((n_steps + 1, 2), dtype=float)
    thetas = np.zeros((n_steps + 1, 2), dtype=float)
    traj[0] = x
    thetas[0] = theta

    for n in range(n_steps):
        _, s, H = mixture_logp_grad_hess(x, comps)
        s = s[0]
        H = H[0]

        if adaptive:
            # ∇ log π_θ(x) = ∇ log π(x) - α ∇^2 log π(x) θ
            s_eff = s - alpha * (H @ theta)
        else:
            s_eff = s

        x_new = x + eps * s_eff + np.sqrt(2 * eps) * rng.normal(size=2)

        if adaptive:
            # Update θ using the NEW state's score (EMA for smooth visuals).
            _, s_new, _ = mixture_logp_grad_hess(x_new, comps)
            #theta = theta + gamma * (s_new[0] - theta)
            theta = theta + 0.1 * (n+2)**(-0.6) * (s_new[0] - theta)

        x = x_new
        traj[n + 1] = x
        thetas[n + 1] = theta

    return traj, thetas


# ---------- Plot helpers ----------

def clip_vector_field(Ux: np.ndarray, Uy: np.ndarray, max_norm: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip vector magnitudes to max_norm (preserve direction)."""
    norm = np.sqrt(Ux * Ux + Uy * Uy)
    scale = np.ones_like(norm)
    mask = norm > max_norm
    scale[mask] = max_norm / (norm[mask] + 1e-12)
    return Ux * scale, Uy * scale


def main() -> None:
    # --- Simulation hyperparameters (chosen for clear visuals, not for best mixing) ---
    n_steps = 4000
    eps = 0.01
    alpha = 1.53 # 3.0
    gamma = 0.01
    seed = 1

    traj_base, _ = simulate_langevin(n_steps, eps, seed=seed, adaptive=False, alpha=alpha, gamma=gamma)
    traj_adap, thetas = simulate_langevin(n_steps, eps, seed=seed, adaptive=True, alpha=alpha, gamma=gamma)

    # --- Plotting grids ---
    xlim = (-4.0, 4.5)
    ylim = (-3.5, 3.5)

    # Fine grid for heatmaps
    nx_f, ny_f = 220, 200
    xs_f = np.linspace(*xlim, nx_f)
    ys_f = np.linspace(*ylim, ny_f)
    Xf, Yf = np.meshgrid(xs_f, ys_f)
    grid_f = np.stack([Xf.ravel(), Yf.ravel()], axis=1)

    logp_f, score_f, _ = mixture_logp_grad_hess(grid_f, comps)
    U_f = (-logp_f).reshape(ny_f, nx_f)

    # Coarse grid for quiver arrows
    nx_q, ny_q = 11, 10 # original value: 19, 17
    xs_q = np.linspace(*xlim, nx_q)
    ys_q = np.linspace(*ylim, ny_q)
    Xq, Yq = np.meshgrid(xs_q, ys_q)
    grid_q = np.stack([Xq.ravel(), Yq.ravel()], axis=1)

    logp_q, score_q, hess_q = mixture_logp_grad_hess(grid_q, comps)
    Sxq = score_q[:, 0].reshape(ny_q, nx_q)
    Syq = score_q[:, 1].reshape(ny_q, nx_q)

    # --- Figure: 2 rows x 4 columns snapshots ---
    snapshots = [0, 1000, 2000, 3000]
    fig, axes = plt.subplots(2, len(snapshots), figsize=(14, 6), constrained_layout=True)

    # Color scaling (percentile-based, fixed across panels)
    top_vmin, top_vmax = np.percentile(U_f, 2.0), np.percentile(U_f, 98.0)

    # Precompute bottom-row potentials to set a global color scale
    Utheta_list = []
    for t in snapshots:
        theta_t = thetas[t]
        # log π_θ = log π - α θ^T s(x)  =>  U_θ = -log π_θ = U + α θ^T s(x)
        Utheta = (-logp_f - (-alpha * (score_f @ theta_t))).reshape(ny_f, nx_f)
        # equivalent: U_f + alpha*(score_f@theta_t)
        Utheta_list.append(Utheta)
    Utheta_stack = np.stack(Utheta_list, axis=0)
    bot_vmin, bot_vmax = np.percentile(Utheta_stack, 2.0), np.percentile(Utheta_stack, 98.0)

    # Quiver styling
    q_scale = 10.0
    max_norm = 7.0

    for j, t in enumerate(snapshots):
        # --- Top row: baseline (fixed field) ---
        ax = axes[0, j]
        ax.imshow(
            U_f,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            aspect="auto",
            vmin=top_vmin,
            vmax=top_vmax,
        )

        ax.plot(traj_base[: t + 1, 0], traj_base[: t + 1, 1], linewidth=1.0, color="white")
        ax.scatter([traj_base[t, 0]], [traj_base[t, 1]], s=15, zorder=5, c='cyan')

        Ux_disp, Uy_disp = clip_vector_field(Sxq, Syq, max_norm=max_norm)
        ax.quiver(
            Xq,
            Yq,
            Ux_disp,
            Uy_disp,
            angles="xy",
            scale_units="xy",
            scale=q_scale,
            width=0.0045,
            headwidth=3.5,
            headlength=5.0,
            headaxislength=4.5,
            color="0.75",
            alpha=0.95,
            zorder=3,
        )

        ax.set_title(f"n = {t}", fontsize=16)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Overdamped Langevin\n$\\nabla \\log \\pi(x)$", fontsize=16)

        # --- Bottom row: adaptive (time-varying field + time-varying potential) ---
        ax = axes[1, j]
        theta_t = thetas[t]

        # Heatmap of adapted potential U_θ
        Utheta = Utheta_list[j]
        ax.imshow(
            Utheta,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            aspect="auto",
            vmin=bot_vmin,
            vmax=bot_vmax,
        )

        ax.plot(traj_adap[: t + 1, 0], traj_adap[: t + 1, 1], linewidth=1.0, color="white")
        ax.scatter([traj_adap[t, 0]], [traj_adap[t, 1]], s=15, zorder=5, c='cyan')

        # Adapted score field: s_θ(x) = s(x) - α H_logπ(x) θ_t
        s_theta_q = score_q - alpha * np.einsum("nij,j->ni", hess_q, theta_t)
        Sxtq = s_theta_q[:, 0].reshape(ny_q, nx_q)
        Sytq = s_theta_q[:, 1].reshape(ny_q, nx_q)

        # Reference (baseline) arrows in faint gray
        Ux0, Uy0 = clip_vector_field(Sxq, Syq, max_norm=max_norm)
        ax.quiver(
            Xq,
            Yq,
            Ux0,
            Uy0,
            angles="xy",
            scale_units="xy",
            scale=q_scale,
            width=0.0025,
            headwidth=3.2,
            headlength=4.5,
            headaxislength=4.0,
            color="0.75",
            alpha=0.35,
            zorder=3,
        )

        # Adapted arrows in orange/red
        Ux1, Uy1 = clip_vector_field(Sxtq, Sytq, max_norm=max_norm)
        ax.quiver(
            Xq,
            Yq,
            Ux1,
            Uy1,
            angles="xy",
            scale_units="xy",
            scale=q_scale,
            width=0.0045,
            headwidth=3.8,
            headlength=5.5,
            headaxislength=5.0,
            color="orangered",
            alpha=0.95,
            zorder=3,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Score-Repellent MC\n$\\nabla \\log \\pi_{\\theta_n}(x)$", fontsize=16)

    out_pdf = Path("./score_based_mcmc_intro_figure.pdf")
    out_png = Path("./score_based_mcmc_intro_figure.png")

    # Add legend below the plot
    handles = [
        mlines.Line2D([0], [0], color="0.75", marker=r'$\longrightarrow$', markersize=10, linestyle='none', label="Original score"),
        mlines.Line2D([0], [0], color="orangered", marker=r'$\longrightarrow$', markersize=10, linestyle='none', label="Adaptive score"),
        mlines.Line2D([0], [0], marker='o', color='cyan', markersize=4, linestyle='none', label="Current sample")
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=16)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
