"""
animate.py
──────────
Animated visualization of offloading over time for each policy.

Usage
-----
    python animate.py                   # default: PD, R=4, onoff
    python animate.py --policy LPF --R 6 --channel general
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from config import (
    AP_POSITIONS, AP_RANGE, NUM_USERS, NUM_STATIONARY, NUM_MOBILE,
    MARKER_STATIONARY, MARKER_MOBILE,
    COLOR_IDLE, COLOR_SERVING, COLOR_DONE, COLOR_CELLULAR,
    MOBILE_AREA_HALF,
)
from environment import Environment
from policies import POLICY_FNS


def _user_states(delivered, Ci, Ti, t, serving_set):
    """Classify each user into one of four states."""
    colors = np.full(NUM_USERS, COLOR_IDLE, dtype=object)
    for i in range(NUM_USERS):
        if delivered[i] >= Ci[i] - 1e-9:
            colors[i] = COLOR_DONE
        elif Ti[i] <= t:
            colors[i] = COLOR_CELLULAR
        # else: IDLE (default)
    return colors


def generate_animation(policy_name="PD", R=4, channel="onoff",
                       seed=0, fps=30, skip=10, out_dir="results"):
    """
    Run one policy with logging, then render an MP4.

    Parameters
    ----------
    skip : int
        Render every `skip`-th frame to keep video length reasonable.
    """
    env = Environment(channel, seed=seed)
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max

    print(f"Running {policy_name} (R={R}, {channel}, seed={seed}) with logging ...")
    fn = POLICY_FNS[policy_name]
    total_off, log = fn(env, R=R, record_log=True)
    demand = Ci.sum()
    print(f"  Offload ratio: {total_off / demand:.4f}  ({len(log)} slots logged)")

    frames = list(range(0, len(log), skip))
    if frames[-1] != len(log) - 1:
        frames.append(len(log) - 1)
    n_frames = len(frames)
    duration_s = n_frames / fps
    print(f"  Rendering {n_frames} frames @ {fps} fps  ({duration_s:.1f}s video)")

    # ── figure setup ──
    fig, ax = plt.subplots(figsize=(8, 8))
    pad = 200
    lim = MOBILE_AREA_HALF + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    for pos in AP_POSITIONS:
        circ = Circle(pos, AP_RANGE, fill=False, edgecolor="black",
                      linewidth=0.8, linestyle="--", alpha=0.5)
        ax.add_patch(circ)
        ax.plot(*pos, "k+", markersize=8, markeredgewidth=1.5)

    stat_scatter = ax.scatter([], [], marker=MARKER_STATIONARY, s=30,
                              edgecolors="black", linewidths=0.3, zorder=3)
    mob_scatter = ax.scatter([], [], marker=MARKER_MOBILE, s=30,
                             edgecolors="black", linewidths=0.3, zorder=3)

    title = ax.set_title("", fontsize=12)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_IDLE,
                   markersize=8, label="Idle"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DONE,
                   markersize=8, label="Done"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CELLULAR,
                   markersize=8, label="Cellular"),
        plt.Line2D([0], [0], marker=MARKER_STATIONARY, color="w",
                   markerfacecolor="gray", markersize=8, label="Stationary"),
        plt.Line2D([0], [0], marker=MARKER_MOBILE, color="w",
                   markerfacecolor="gray", markersize=8, label="Mobile"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9)

    stats_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                         fontsize=9, verticalalignment="bottom",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def _update(frame_idx):
        fi = frames[frame_idx]
        entry = log[fi]
        t = fi + 1
        pos = entry["positions"]
        delivered = entry["delivered"]
        serving_set = set(u for u, _ in entry["serving"])

        colors = _user_states(delivered, Ci, Ti, t, serving_set)

        stat_pos = pos[:NUM_STATIONARY]
        mob_pos = pos[NUM_STATIONARY:]
        stat_colors = colors[:NUM_STATIONARY]
        mob_colors = colors[NUM_STATIONARY:]

        stat_scatter.set_offsets(stat_pos)
        stat_scatter.set_facecolors(stat_colors)
        mob_scatter.set_offsets(mob_pos)
        mob_scatter.set_facecolors(mob_colors)

        n_done = int(np.sum(delivered >= Ci - 1e-9))
        n_cell = int(np.sum((delivered < Ci - 1e-9) & (Ti <= t)))
        ratio = float(np.minimum(delivered, Ci).sum()) / demand

        title.set_text(f"{policy_name}  R={R}  {channel}  |  t = {t}/{T_max}")
        stats_text.set_text(
            f"Offload: {ratio:.1%}\n"
            f"Done: {n_done}/{NUM_USERS}  |  Cellular: {n_cell}"
        )
        return stat_scatter, mob_scatter, title, stats_text

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                   interval=1000 // fps, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"anim_{policy_name}_R{R}_{channel}.mp4"
    out_path = os.path.join(out_dir, fname)

    writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate offloading")
    parser.add_argument("--policy", default="PD", choices=["RR", "MW", "PD", "LPF"])
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--channel", default="onoff", choices=["onoff", "general"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=50,
                        help="render every N-th slot (default 50)")
    args = parser.parse_args()

    generate_animation(
        policy_name=args.policy,
        R=args.R,
        channel=args.channel,
        seed=args.seed,
        fps=args.fps,
        skip=args.skip,
    )
