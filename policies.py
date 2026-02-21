"""
policies.py
───────────
Five scheduling policies: RR, MW, PF, PD, LPF.

Every ``run_*`` function has the same signature:

    run_*(env, R, record_log=False)  →  (total_offloaded: float, log: list|None)

``log`` (if requested) is a list of dicts, one per time slot:
    {"positions": ndarray(N,2), "delivered": ndarray(N,), "serving": [(user,ap), ...]}
"""

import numpy as np
from config import NUM_USERS, NUM_APS


# ───────────────────────────────────────────────
#  helpers
# ───────────────────────────────────────────────
def _log_entry(pos, delivered, serving):
    return {
        "positions": pos.copy(),
        "delivered": delivered.copy(),
        "serving": list(serving),
    }


def _all_done(delivered, Ci, Ti, t):
    """True when every user is either fully served or past deadline."""
    return np.all((delivered >= Ci - 1e-9) | (Ti <= t))


# ═══════════════════════════════════════════════
#  Round Robin  (RR)
# ═══════════════════════════════════════════════
def run_rr(env, R, record_log=False):
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max
    delivered = np.zeros(NUM_USERS)
    log = [] if record_log else None

    for t in range(1, T_max + 1):
        K, pos = env.get_state(t)
        data_slot = np.zeros(NUM_USERS)
        serving = []

        for m in range(NUM_APS):
            elig = (K[:, m] > 0) & (delivered < Ci)
            n = elig.sum()
            if n == 0:
                continue
            x_each = R / n
            data_slot[elig] += x_each * K[elig, m]
            if record_log:
                serving.extend((int(i), m) for i in np.where(elig)[0])

        remaining = np.maximum(Ci - delivered, 0.0)
        delivered += np.minimum(data_slot, remaining)

        if record_log:
            log.append(_log_entry(pos, delivered, serving))
        if _all_done(delivered, Ci, Ti, t):
            break

    return float(np.minimum(delivered, Ci).sum()), log


# ═══════════════════════════════════════════════
#  Max-Weight  (MW)
# ═══════════════════════════════════════════════
def run_mw(env, R, record_log=False):
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max
    delivered = np.zeros(NUM_USERS)
    log = [] if record_log else None

    for t in range(1, T_max + 1):
        K, pos = env.get_state(t)
        data_slot = np.zeros(NUM_USERS)
        serving = []

        for m in range(NUM_APS):
            elig = (K[:, m] > 0) & (delivered < Ci)
            idx = np.where(elig)[0]
            if len(idx) == 0:
                continue
            scores = K[idx, m] * (Ci[idx] - delivered[idx])
            i_star = idx[np.argmax(scores)]
            data_slot[i_star] += R * K[i_star, m]
            serving.append((int(i_star), m))

        remaining = np.maximum(Ci - delivered, 0.0)
        delivered += np.minimum(data_slot, remaining)

        if record_log:
            log.append(_log_entry(pos, delivered, serving))
        if _all_done(delivered, Ci, Ti, t):
            break

    return float(np.minimum(delivered, Ci).sum()), log


# ═══════════════════════════════════════════════
#  Proportional Fair  (PF)
# ═══════════════════════════════════════════════
def run_pf(env, R, record_log=False):
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max
    delivered = np.zeros(NUM_USERS)
    log = [] if record_log else None

    for t in range(1, T_max + 1):
        K, pos = env.get_state(t)
        data_slot = np.zeros(NUM_USERS)
        serving = []

        throughput = delivered / max(t - 1, 1)

        for m in range(NUM_APS):
            elig = (K[:, m] > 0) & (delivered < Ci)
            idx = np.where(elig)[0]
            if len(idx) == 0:
                continue
            tp = np.where(throughput[idx] > 0, throughput[idx], 1e-12)
            scores = K[idx, m] / tp
            i_star = idx[np.argmax(scores)]
            data_slot[i_star] += R * K[i_star, m]
            serving.append((int(i_star), m))

        remaining = np.maximum(Ci - delivered, 0.0)
        delivered += np.minimum(data_slot, remaining)

        if record_log:
            log.append(_log_entry(pos, delivered, serving))
        if _all_done(delivered, Ci, Ti, t):
            break

    return float(np.minimum(delivered, Ci).sum()), log


# ═══════════════════════════════════════════════
#  Primal-Dual  (PD) — Algorithm 2
# ═══════════════════════════════════════════════
def run_pd(env, R, record_log=False):
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max
    delivered = np.zeros(NUM_USERS)
    Z = np.zeros(NUM_USERS)
    Cmin = Ci.min()
    d = (1.0 + 1.0 / Cmin) ** (Cmin / R)
    log = [] if record_log else None

    for t in range(1, T_max + 1):
        K, pos = env.get_state(t)
        serving = []

        served_aps = {}                       # user_idx → [ap_idx, ...]
        for m in range(NUM_APS):
            elig = (K[:, m] > 0) & (delivered < Ci)
            idx = np.where(elig)[0]
            if len(idx) == 0:
                continue
            scores = K[idx, m] * (1.0 - Z[idx])
            best = np.argmax(scores)
            if scores[best] <= 0:
                continue
            i_star = idx[best]
            served_aps.setdefault(int(i_star), []).append(m)
            serving.append((int(i_star), m))

        for i, aps in served_aps.items():
            sum_K = K[i, aps].sum()
            Z[i] = Z[i] * (1.0 + sum_K / Ci[i]) + sum_K / ((d - 1.0) * Ci[i])
            data = R * sum_K
            rem = Ci[i] - delivered[i]
            delivered[i] += min(data, rem)

        if record_log:
            log.append(_log_entry(pos, delivered, serving))
        if _all_done(delivered, Ci, Ti, t):
            break

    return float(np.minimum(delivered, Ci).sum()), log


# ═══════════════════════════════════════════════
#  Least Progress First  (LPF) — Algorithm 3
# ═══════════════════════════════════════════════
def run_lpf(env, R, record_log=False):
    Ci, Ti, T_max = env.Ci, env.Ti, env.T_max
    delivered = np.zeros(NUM_USERS)
    log = [] if record_log else None

    for t in range(1, T_max + 1):
        K, pos = env.get_state(t)
        data_slot = np.zeros(NUM_USERS)
        serving = []

        # score = K_imt × (undelivered / Ci)
        unfrac = (Ci - delivered) / Ci

        for m in range(NUM_APS):
            elig = (K[:, m] > 0) & (delivered < Ci)
            idx = np.where(elig)[0]
            if len(idx) == 0:
                continue
            scores = K[idx, m] * unfrac[idx]
            i_star = idx[np.argmax(scores)]
            data_slot[i_star] += R * K[i_star, m]
            serving.append((int(i_star), m))

        remaining = np.maximum(Ci - delivered, 0.0)
        delivered += np.minimum(data_slot, remaining)

        if record_log:
            log.append(_log_entry(pos, delivered, serving))
        if _all_done(delivered, Ci, Ti, t):
            break

    return float(np.minimum(delivered, Ci).sum()), log


# ═══════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════
POLICY_FNS = {
    "RR":  run_rr,
    "MW":  run_mw,
    "PF":  run_pf,
    "PD":  run_pd,
    "LPF": run_lpf,
}
