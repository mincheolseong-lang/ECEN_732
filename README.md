# ECEN_732 — Mideterm presentation

Implementation of the paper:

**"On the Capacity-Performance Trade-off of Online Policy in Delayed Mobile Offloading"**

---

## Requirements

- **Python 3.7+**
- **Libraries:** `numpy`, `scipy`, `matplotlib`  
  (Optional: `ffmpeg` for generating animation videos via `animate.py`)

### Install (pip)

```bash
pip install numpy scipy matplotlib
```

### Install (conda, recommended)

```bash
conda create -n ecen732 python=3.10
conda activate ecen732
conda install numpy scipy matplotlib
```

---

## How to Run

1. Install the required libraries (see above).
2. From the project root:

```bash
python main.py
```

This will:

- Generate Figure 1 (approximation), Figure 3 (capacity), Figure 4 (AP & client locations).
- Run general-channel simulations (Fig. 6) and save results under `results/`.

Outputs are written to the `results/` directory (plots and, if saved, `.npz` data).

---

## Optional: Animation

To generate an MP4 of the offloading process over time:

```bash
python animate.py --policy PD --R 4 --channel general
```

Requires `ffmpeg` for encoding. See `python animate.py --help` for options.

---

## Configuration

Main parameters are in `config.py`:

- `NUM_STATIONARY`, `NUM_MOBILE` — number of stationary/mobile users
- `LIGHT_RATIO` — fraction of light users per group (e.g. 0.95 = 95% light, 5% heavy)
- `MOBILE_AREA_HALF` — mobile users sampled in `[-HALF, HALF]²`
- `R_VALUES`, `NUM_RUNS` — AP capacity values and number of seeds for averaging
