"""
SafeTwin / FROST-Twin pilot experiments (E0-E5).
Self-contained ring-road IDM proxy pilot -- see experiment_plan.md for full rationale.
Designed to run top-to-bottom in <45 min on Kaggle CPU (torch used for small models only).
"""

import json
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get("PILOT_OUT_DIR", "/kaggle/working")
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 0
np.random.seed(RNG_SEED)
if torch is not None:
    torch.manual_seed(RNG_SEED)

SUMMARY = {}


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved figure:", path)


# ---------------------------------------------------------------------------
# 0. Shared substrate: ring-road IDM microsimulator
# ---------------------------------------------------------------------------

def idm_accel(v, v_lead, gap, v0, T, a, b, s0, friction=1.0):
    """Intelligent Driver Model acceleration. `friction` in (0,1] scales max accel/decel
    (a crude proxy for the memos' weather-dependent friction coefficient mu)."""
    a_eff = a * friction
    b_eff = b * friction
    gap = max(gap, 1e-3)
    s_star = s0 + max(0.0, v * T + v * (v - v_lead) / (2.0 * math.sqrt(a_eff * b_eff)))
    accel = a_eff * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2)
    return accel


def simulate_ring(
    N, L, v0, T, a, b, s0, friction=1.0, steps=140, dt=0.5,
    cav_idx=None, cav_accel_fn=None, rng=None,
):
    """Simulate a single-lane ring road of length L with N vehicles.
    Returns dict with per-step mean speed, std speed, and full speed history.
    If cav_idx is given, that vehicle's acceleration is overridden by cav_accel_fn(t, obs_dict).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    positions = np.linspace(0, L, N, endpoint=False)
    positions = positions + rng.normal(0, 0.15, size=N)
    positions = np.mod(positions, L)
    order = np.argsort(positions)
    positions = positions[order]
    velocities = np.full(N, v0 * 0.8) + rng.normal(0, 0.3, size=N)
    velocities = np.clip(velocities, 0.5, v0)

    speed_hist = np.zeros((steps, N))
    headway_violations = 0

    for t in range(steps):
        idx_sorted = np.argsort(positions)
        pos_sorted = positions[idx_sorted]
        vel_sorted = velocities[idx_sorted]
        lead = np.roll(np.arange(N), -1)
        gaps = np.mod(pos_sorted[lead] - pos_sorted, L)
        gaps[gaps <= 0] += L

        accels_sorted = np.zeros(N)
        for i in range(N):
            v = vel_sorted[i]
            v_lead = vel_sorted[lead[i]]
            gap = gaps[i]
            accels_sorted[i] = idm_accel(v, v_lead, gap, v0, T, a, b, s0, friction=friction)

        if cav_idx is not None and cav_accel_fn is not None:
            rank_of_cav = np.where(idx_sorted == cav_idx)[0][0]
            obs = {
                "v": vel_sorted[rank_of_cav],
                "v_lead": vel_sorted[lead[rank_of_cav]],
                "gap": gaps[rank_of_cav],
                "t": t,
            }
            accels_sorted[rank_of_cav] = cav_accel_fn(obs)

        vel_sorted = np.clip(vel_sorted + accels_sorted * dt, 0.0, v0 * 1.3)
        pos_sorted = np.mod(pos_sorted + vel_sorted * dt, L)

        min_headway = np.min(gaps / np.maximum(vel_sorted, 1e-2))
        if min_headway < 0.5:
            headway_violations += 1

        inv_sort = np.argsort(idx_sorted)
        positions = pos_sorted[inv_sort]
        velocities = vel_sorted[inv_sort]
        speed_hist[t] = velocities

    mean_speed = speed_hist.mean(axis=1)
    std_speed = speed_hist.std(axis=1)
    return {
        "speed_hist": speed_hist,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "headway_violations": headway_violations,
    }


print("=" * 70)
print("Ring-road substrate ready.")
