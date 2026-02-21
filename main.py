"""
main.py
───────
Entry point — reproduce Figures 1, 3, 4, 5, 6 from the paper.

Usage
-----
    conda activate ecen732
    cd ~/ECEN_732
    python main.py
"""

import os
import numpy as np
from environment import Environment
from simulate import run_experiment
from plot_results import plot_fig1, plot_fig3, plot_fig4, plot_performance

OUT_DIR = "results"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Figure 1 & 3  (analytical, instant) ───────────────
    print("=" * 50)
    print("Generating Figure 1 (approximation) ...")
    plot_fig1(save_path=os.path.join(OUT_DIR, "fig1_approximation.png"))

    print("Generating Figure 3 (capacity requirements) ...")
    plot_fig3(save_path=os.path.join(OUT_DIR, "fig3_capacity.png"))

    # ── Figure 4  (location snapshot) ─────────────────────
    print("=" * 50)
    print("Generating Figure 4 (AP & client locations) ...")
    env_snap = Environment(channel_type="onoff", seed=0)
    plot_fig4(env_snap, save_path=os.path.join(OUT_DIR, "fig4_locations.png"))

    # ── Figure 6  (General channels) ─────────────────────
    print("=" * 50)
    print("Running General channel simulations ...")
    ratios_gen, stds_gen = run_experiment("general", verbose=True)
    plot_performance(
        ratios_gen, "general",
        save_path=os.path.join(OUT_DIR, "fig6_general.png"),
    )
    np.savez(
        os.path.join(OUT_DIR, "data_general.npz"),
        **{f"ratio_{n}": ratios_gen[n] for n in ratios_gen},
        **{f"std_{n}": stds_gen[n] for n in stds_gen},
    )

    print("=" * 50)
    print("Done!  Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
