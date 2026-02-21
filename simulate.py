"""
simulate.py
───────────
Run all policies across multiple R values and seeds,
return a results dict ready for plotting.
"""

import time
import numpy as np
from environment import Environment
from policies import POLICY_FNS
from config import R_VALUES, NUM_RUNS, POLICY_NAMES


def run_single(channel_type, R, seed, record_log=False):
    """
    Run every policy on one (channel_type, R, seed) instance.

    Returns
    -------
    results : dict   policy_name → total_offloaded (float)
    logs    : dict   policy_name → log list  (only if record_log)
    total_demand : float
    """
    env = Environment(channel_type=channel_type, seed=seed)
    total_demand = env.Ci.sum()

    results = {}
    logs = {} if record_log else None

    for name in POLICY_NAMES:
        offloaded, log = POLICY_FNS[name](env, R, record_log=record_log)
        results[name] = offloaded
        if record_log:
            logs[name] = log

    return results, logs, total_demand


def run_experiment(channel_type, seeds=None, verbose=True):
    """
    Full experiment: all R values × all seeds.

    Parameters
    ----------
    channel_type : ``"onoff"`` | ``"general"``
    seeds        : list[int] | None   (None → 0, 1, …, NUM_RUNS-1)
    verbose      : bool

    Returns
    -------
    ratios : dict   policy_name → ndarray(len(R_VALUES),)
        Average offload ratio over seeds.
    stds   : dict   policy_name → ndarray(len(R_VALUES),)
        Std-dev of offload ratio over seeds.
    """
    if seeds is None:
        seeds = list(range(NUM_RUNS))

    n_R = len(R_VALUES)
    n_seeds = len(seeds)

    raw = {name: np.zeros((n_R, n_seeds)) for name in POLICY_NAMES}

    for ri, R in enumerate(R_VALUES):
        for si, seed in enumerate(seeds):
            t0 = time.time()
            res, _, demand = run_single(channel_type, R, seed)
            elapsed = time.time() - t0

            for name in POLICY_NAMES:
                raw[name][ri, si] = res[name] / demand

            if verbose:
                tag = ", ".join(
                    f"{n}={raw[n][ri, si]:.4f}" for n in POLICY_NAMES
                )
                print(f"[{channel_type}] R={R}  seed={seed}  "
                      f"({elapsed:.1f}s)  {tag}")

    ratios = {n: raw[n].mean(axis=1) for n in POLICY_NAMES}
    stds = {n: raw[n].std(axis=1) for n in POLICY_NAMES}

    return ratios, stds
