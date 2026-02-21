import numpy as np

# ═══════════════════════════════════════════════════════════
#  AP Configuration
# ═══════════════════════════════════════════════════════════
NUM_APS = 9
AP_GRID_SIZE = 3                        # 3 x 3 grid
AP_SPACING = 1000.0                     # meters between adjacent APs
AP_RANGE = 400.0                        # transmission range (meters)

AP_POSITIONS = np.array([
    ((r - 1) * AP_SPACING, (c - 1) * AP_SPACING)
    for r in range(AP_GRID_SIZE)
    for c in range(AP_GRID_SIZE)
])  # 9 APs centered at origin: (-1000,-1000) to (1000,1000)

# ═══════════════════════════════════════════════════════════
#  User Configuration
# ═══════════════════════════════════════════════════════════
NUM_STATIONARY = 100
NUM_MOBILE = 100
NUM_USERS = NUM_STATIONARY + NUM_MOBILE

LIGHT_RATIO = 0.85                      # 85% light, 15% heavy in each group

C_LIGHT = 100                           # data amount for light users
C_HEAVY = 10_000                        # data amount for heavy users

# ═══════════════════════════════════════════════════════════
#  Channel Model
# ═══════════════════════════════════════════════════════════
PATHLOSS_REF = 80.0                     # reference distance for pathloss
ONOFF_THRESHOLD = 1.0 / 25.0           # On-Off channel threshold
FADING_VAR = 1.0                        # per-component Gaussian variance
                                        # N(0,1) as stated in paper

# ═══════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════
R_VALUES = list(range(1, 10))            # AP capacity R = 1, 2, ..., 9
#R_VALUES = [6]
NUM_RUNS = 5                           # quick test (full: 5)

# ═══════════════════════════════════════════════════════════
#  Geometry (mobile user sampling area)
# ═══════════════════════════════════════════════════════════
MOBILE_AREA_HALF = 1500.0               # mobile users in [-1500, 1500]^2

# ═══════════════════════════════════════════════════════════
#  Policy Names
# ═══════════════════════════════════════════════════════════
POLICY_NAMES = ["RR", "MW", "PD", "LPF"]

# ═══════════════════════════════════════════════════════════
#  Visualization (animation markers & colors)
# ═══════════════════════════════════════════════════════════
MARKER_STATIONARY = "o"                 # circle for stationary users
MARKER_MOBILE = "^"                     # triangle for mobile users

COLOR_IDLE = "#AAAAAA"                  # gray:  not being served
COLOR_SERVING = "#2196F3"               # blue:  currently receiving data
COLOR_DONE = "#4CAF50"                  # green: offloading complete
COLOR_CELLULAR = "#F44336"              # red:   deadline passed → cellular
