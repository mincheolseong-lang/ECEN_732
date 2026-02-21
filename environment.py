"""
environment.py
──────────────
AP layout, user placement, channel model (K_imt).
All randomness is seeded so every policy sees identical channels.
"""

import numpy as np
from config import (
    AP_POSITIONS, AP_RANGE,
    NUM_STATIONARY, NUM_MOBILE, NUM_USERS, NUM_APS,
    LIGHT_RATIO, C_LIGHT, C_HEAVY,
    PATHLOSS_REF, ONOFF_THRESHOLD, MOBILE_AREA_HALF,
    FADING_VAR,
)


# ═══════════════════════════════════════════════════════════
#  User data (Ci, Ti) — same for every run
# ═══════════════════════════════════════════════════════════
def make_user_params():
    """
    Build Ci (data demand) and Ti (deadline) for all users.

    Within each group, 95% are light, 5% are heavy:
      light : Ci = 100,   Ti = 50 + 50*i
      heavy : Ci = 10000, Ti = 2500*(i - n_light)
    """
    Ci = np.zeros(NUM_USERS)
    Ti = np.zeros(NUM_USERS, dtype=int)

    for group_size, group_offset in [(NUM_STATIONARY, 0),
                                     (NUM_MOBILE, NUM_STATIONARY)]:
        n_light = int(group_size * LIGHT_RATIO)
        for j in range(group_size):
            idx = group_offset + j
            n = j + 1
            if n <= n_light:
                Ci[idx] = C_LIGHT
                Ti[idx] = 50 + 50 * n
            else:
                Ci[idx] = C_HEAVY
                Ti[idx] = 2500 * (n - n_light)

    return Ci, Ti


# ═══════════════════════════════════════════════════════════
#  Environment class
# ═══════════════════════════════════════════════════════════
class Environment:
    """
    One simulation instance.
    Call ``get_state(t)`` to obtain the channel matrix and positions
    for any time slot.  Results are deterministic for a given seed,
    regardless of how many times or in what order you call get_state.
    """

    def __init__(self, channel_type="onoff", seed=None):
        """
        Parameters
        ----------
        channel_type : ``"onoff"`` | ``"general"``
        seed : int or None
        """
        self.channel_type = channel_type
        master = np.random.default_rng(seed)

        # user parameters
        self.Ci, self.Ti = make_user_params()
        self.T_max = 12500

        # boolean mask: True for mobile users (indices 100-199)
        self.is_mobile = np.concatenate([
            np.zeros(NUM_STATIONARY, dtype=bool),
            np.ones(NUM_MOBILE, dtype=bool),
        ])

        # stationary positions (fixed for the entire run)
        self.stationary_pos = self._sample_in_coverage(master)

        # one seed per time slot → reproducible across policy runs
        self._step_seeds = master.integers(0, 2**63, size=self.T_max + 1)

    def __repr__(self):
        return (
            f"Environment(type={self.channel_type}, T_max={self.T_max}, "
            f"users={NUM_USERS}, aps={NUM_APS})"
        )

    # ──────────────────────────────────────────────
    #  Stationary user placement (rejection sampling)
    # ──────────────────────────────────────────────
    def _sample_in_coverage(self, rng):
        """Uniform random within the union of 9 AP coverage circles."""
        lo = AP_POSITIONS.min(axis=0) - AP_RANGE
        hi = AP_POSITIONS.max(axis=0) + AP_RANGE

        pts = np.empty((0, 2))
        while pts.shape[0] < NUM_STATIONARY:
            xy = rng.uniform(lo, hi, (NUM_STATIONARY * 4, 2))
            dists = np.linalg.norm(
                xy[:, None, :] - AP_POSITIONS[None, :, :], axis=2
            )
            pts = np.vstack([pts, xy[np.any(dists <= AP_RANGE, axis=1)]])

        return pts[:NUM_STATIONARY]

    # ──────────────────────────────────────────────
    #  Per-slot channel matrix
    # ──────────────────────────────────────────────
    def get_state(self, t):
        """
        Parameters
        ----------
        t : int  (1-indexed time slot)

        Returns
        -------
        K         : ndarray (NUM_USERS, NUM_APS)   channel capacities
        positions : ndarray (NUM_USERS, 2)          user [x, y]
        """
        rng = np.random.default_rng(self._step_seeds[t])

        # ── user positions ──
        positions = np.empty((NUM_USERS, 2))
        positions[:NUM_STATIONARY] = self.stationary_pos
        positions[NUM_STATIONARY:] = rng.uniform(
            -MOBILE_AREA_HALF, MOBILE_AREA_HALF, (NUM_MOBILE, 2)
        )

        # ── distances  (N_users × N_aps) ──
        diff = positions[:, None, :] - AP_POSITIONS[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # ── pathloss: min{1, 1/(d/80)^2} ──
        with np.errstate(divide="ignore"):
            pl = np.minimum(1.0, 1.0 / (dist / PATHLOSS_REF) ** 2)
        pl = np.where(np.isfinite(pl), pl, 1.0)

        # ── Rayleigh fading: sqrt(a^2 + b^2),  a,b ~ N(0,σ²) ──
        # Paper says "variance 1", but standard wireless convention
        # uses E[|h|^2]=1 (power-normalized), giving σ²=0.5 per component.
        # Toggle FADING_VAR in config.py to test both.
        a = rng.standard_normal((NUM_USERS, NUM_APS))
        b = rng.standard_normal((NUM_USERS, NUM_APS))
        fading = np.sqrt((a ** 2 + b ** 2) * FADING_VAR)

        # ── raw channel gain ──
        gain = pl * fading

        # ── cutoffs ──
        gain[dist > AP_RANGE] = 0.0             # out of range
        gain[self.Ti < t] = 0.0                 # past deadline (row-wise)

        # ── map to K_imt ──
        if self.channel_type == "onoff":
            K = np.where(gain > ONOFF_THRESHOLD, 1.0, 0.0)
        else:  # general
            K = np.minimum(gain, 1.0)

        return K, positions
