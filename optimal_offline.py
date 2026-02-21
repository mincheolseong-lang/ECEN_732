"""
optimal_offline.py
──────────────────
Solve the Offload LP for Γ_opt (optimal offline offloaded data).

    Max  Σ X_imt · K_imt
    s.t. Σ_{m,t} X_imt · K_imt  ≤  Ci      ∀ i       (user cap)
         Σ_i     X_imt           ≤  R       ∀ (m,t)   (AP slot)
         X_imt ≥ 0

NOTE
----
The full 200-user / 25 000-slot instance produces ~400 K+ variables.
HiGHS can handle it, but expect minutes of runtime.
Use ``max_T`` to cap the horizon for quick sanity checks.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from config import NUM_USERS


def solve_offline(env, R=1, max_T=None, time_limit=600, verbose=True):
    """
    Parameters
    ----------
    env        : Environment
    R          : float       capacity multiplier  (default 1)
    max_T      : int|None    only use t = 1 … max_T  (None → full horizon)
    time_limit : float       seconds for HiGHS solver
    verbose    : bool

    Returns
    -------
    gamma_opt : float | None    (None if solver fails)
    """
    Ci = env.Ci
    T = min(env.T_max, max_T) if max_T else env.T_max

    # ── 1. collect non-zero K_imt entries ─────────────────
    if verbose:
        print(f"[offline LP] scanning t = 1 … {T}")

    list_i, list_m, list_t, list_k = [], [], [], []

    for t in range(1, T + 1):
        K, _ = env.get_state(t)
        rows, cols = np.nonzero(K)
        n = len(rows)
        if n == 0:
            continue
        list_i.append(rows)
        list_m.append(cols)
        list_t.append(np.full(n, t, dtype=np.int32))
        list_k.append(K[rows, cols])

    if not list_i:
        return 0.0

    vi = np.concatenate(list_i).astype(np.int32)
    vm = np.concatenate(list_m).astype(np.int32)
    vt = np.concatenate(list_t).astype(np.int32)
    vk = np.concatenate(list_k)
    n_vars = len(vi)

    if verbose:
        print(f"[offline LP] {n_vars} variables")

    # ── 2. (m,t) → unique row index ──────────────────────
    mt_code = vm.astype(np.int64) * (T + 1) + vt.astype(np.int64)
    _, mt_inv = np.unique(mt_code, return_inverse=True)
    n_mt = int(mt_inv.max()) + 1

    # ── 3. build sparse A_ub  (user-cap rows  +  AP-slot rows) ──
    n_rows = NUM_USERS + n_mt
    idx = np.arange(n_vars, dtype=np.int32)

    row = np.concatenate([vi, NUM_USERS + mt_inv])
    col = np.concatenate([idx, idx])
    dat = np.concatenate([vk, np.ones(n_vars)])

    A = csr_matrix((dat, (row, col)), shape=(n_rows, n_vars))

    b_ub = np.empty(n_rows)
    b_ub[:NUM_USERS] = Ci
    b_ub[NUM_USERS:] = R

    # ── 4. solve  (minimize −obj) ─────────────────────────
    if verbose:
        print(f"[offline LP] solving  {n_rows} rows × {n_vars} cols …")

    res = linprog(
        c=-vk,
        A_ub=A,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
        options={"time_limit": time_limit, "presolve": True},
    )

    if res.success:
        gamma = -res.fun
        if verbose:
            print(f"[offline LP] Γ_opt = {gamma:.2f}  "
                  f"(offload ratio = {gamma / Ci.sum():.4f})")
        return gamma

    if verbose:
        print(f"[offline LP] failed: {res.message}")
    return None
