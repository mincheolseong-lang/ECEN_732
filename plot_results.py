"""
plot_results.py
───────────────
Generate Figures 1, 3, 4, 5, 6 from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import (
    AP_POSITIONS, AP_RANGE, R_VALUES, POLICY_NAMES,
    MARKER_STATIONARY, MARKER_MOBILE,
)


# ── plot style (colored, distinct markers) ──────
_MARKERS = {"RR": "+", "MW": "s", "PD": "o", "LPF": "*"}
_LINES   = {"RR": "--", "MW": "--", "PD": "-", "LPF": "-"}
_COLORS  = {"RR": "#1f77b4", "MW": "#ff7f0e", "PD": "#2ca02c", "LPF": "#d62728"}


def _apply_style(ax):
    ax.set_xlabel("AP Capacity R")
    ax.set_ylabel("Offload Ratio")
    ax.set_xticks(R_VALUES)
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3)


# ═══════════════════════════════════════════════════════════
#  Figure 1 – approximation  e^{1/R}/[R(e^{1/R}-1)]  vs  1+1/(2R)
# ═══════════════════════════════════════════════════════════
def plot_fig1(save_path="fig1_approximation.png"):
    R = np.linspace(0.5, 20, 200)
    original = np.exp(1 / R) / (R * (np.exp(1 / R) - 1))
    approx = 1 + 1 / (2 * R)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(R, original, "-", color="#1f77b4", linewidth=1.8, label="original")
    ax.plot(R, approx, "--", color="#d62728", linewidth=1.8, label="approximation")
    ax.set_xlabel("R")
    ax.set_ylabel("Competitive Ratio")
    ax.legend()
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ═══════════════════════════════════════════════════════════
#  Figure 3 – capacity requirements for different β
# ═══════════════════════════════════════════════════════════
def plot_fig3(save_path="fig3_capacity.png"):
    inv_beta = np.linspace(0.80, 0.96, 200)
    beta = 1 / inv_beta

    R_pd = 1 / (2 * (beta - 1))
    R_others = 1 / (beta - 1)
    R_lower = R_pd                      # matching lower bound

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(inv_beta, R_pd, "-", color="#2ca02c", linewidth=1.8, label="PD")
    ax.plot(inv_beta, R_others, "--", color="#ff7f0e", linewidth=1.8, label="RR, MW, PF")
    ax.plot(inv_beta, R_lower, ":", color="#1f77b4", linewidth=1.8, label="lower bound")
    ax.set_xlabel("1/β")
    ax.set_ylabel("Required capacity")
    ax.legend()
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ═══════════════════════════════════════════════════════════
#  Figure 4 – AP & client locations (snapshot)
# ═══════════════════════════════════════════════════════════
def plot_fig4(env, save_path="fig4_locations.png"):
    """Plot AP coverage circles, stationary & mobile user positions."""
    _, pos = env.get_state(1)

    fig, ax = plt.subplots(figsize=(6, 6))

    for ap in AP_POSITIONS:
        circle = plt.Circle(ap, AP_RANGE, fill=False,
                            linestyle="--", color="gray")
        ax.add_patch(circle)
    ax.plot(AP_POSITIONS[:, 0], AP_POSITIONS[:, 1],
            "k^", markersize=8, label="AP")

    stat = ~env.is_mobile
    ax.plot(pos[stat, 0], pos[stat, 1],
            MARKER_STATIONARY, color="blue", markersize=3,
            label="Stationary Clients")
    ax.plot(pos[env.is_mobile, 0], pos[env.is_mobile, 1],
            MARKER_MOBILE, color="red", markersize=3,
            label="Mobile Clients")

    ax.set_xlim(-1600, 1600)
    ax.set_ylim(-1600, 1600)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ═══════════════════════════════════════════════════════════
#  Figures 5 & 6 – performance comparison
# ═══════════════════════════════════════════════════════════
def plot_performance(ratios, channel_label, save_path):
    """
    Parameters
    ----------
    ratios : dict  policy_name → ndarray(len(R_VALUES),)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for name in POLICY_NAMES:
        ax.plot(
            R_VALUES, ratios[name],
            marker=_MARKERS[name],
            linestyle=_LINES[name],
            color=_COLORS[name],
            label=name,
            markersize=7,
            linewidth=1.8,
        )

    _apply_style(ax)
    ax.set_title(f"Performance comparison for {channel_label} channels")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
