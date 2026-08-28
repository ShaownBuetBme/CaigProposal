"""
Runs experiments E0-E5 end to end, saves figures + CSV/JSON results to OUT_DIR.
See experiment_plan.md for full experimental rationale and hypothesis mapping.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.model_selection import ParameterSampler
from scipy.stats import qmc

from pilot_experiments import simulate_ring, savefig, OUT_DIR

t_global_start = time.time()
SUMMARY = {}
rng_global = np.random.default_rng(42)

L = 230.0
V0 = 15.0

# ===========================================================================
# E0 -- validation anchor: reproduce ring-road phase transition
# ===========================================================================
print("\n=== E0: phase transition validation anchor ===")
t0 = time.time()
densities_N = list(range(8, 61, 2))
n_seeds_e0 = 4
e0_rows = []
for N in densities_N:
    stds = []
    for seed in range(n_seeds_e0):
        res = simulate_ring(N=N, L=L, v0=V0, T=1.5, a=1.2, b=2.0, s0=2.0,
                             steps=160, dt=0.5, rng=np.random.default_rng(1000 + seed))
        stds.append(res["std_speed"][-40:].mean())
    e0_rows.append({"N": N, "density": N / L, "tail_std_mean": np.mean(stds), "tail_std_sem": np.std(stds) / np.sqrt(n_seeds_e0)})

e0_df = pd.DataFrame(e0_rows)
e0_df.to_csv(os.path.join(OUT_DIR, "e0_results.csv"), index=False)

crit_idx = e0_df["tail_std_mean"].values.argmax()
crit_density_left = e0_df["density"].values[max(0, crit_idx - 3)]
# onset density: first density where tail_std crosses 10% of peak
peak = e0_df["tail_std_mean"].max()
onset_mask = e0_df["tail_std_mean"] > 0.1 * peak
onset_density = e0_df.loc[onset_mask, "density"].iloc[0]

plt.figure(figsize=(6, 4))
plt.errorbar(e0_df["density"], e0_df["tail_std_mean"], yerr=e0_df["tail_std_sem"], marker="o", capsize=3)
plt.axvline(onset_density, color="red", linestyle="--", label=f"onset density ~{onset_density:.3f}")
plt.xlabel("Vehicle density (veh / m)")
plt.ylabel("Tail velocity std (wave amplitude order parameter)")
plt.title("E0: Ring-road phase transition (validation anchor)")
plt.legend()
savefig("fig_e0_phase_transition.png")

SUMMARY["E0"] = {
    "onset_density": float(onset_density),
    "peak_tail_std": float(peak),
    "runtime_sec": time.time() - t0,
}
print("E0 onset density:", onset_density, "peak:", peak)

# ===========================================================================
# E1 -- learned surrogate (WP1 TwinBoost proxy)
# ===========================================================================
print("\n=== E1: learned surrogate of the microsimulator ===")
t0 = time.time()

N_FIXED = 26  # near-critical density from E0, most informative regime
HORIZONS = [4, 8, 16]  # steps ahead (dt=0.5s -> 2s/4s/8s)


def make_rollout(seed, T, a, b, friction=1.0):
    rng = np.random.default_rng(seed)
    res = simulate_ring(N=N_FIXED, L=L, v0=V0, T=T, a=a, b=b, s0=2.0,
                         friction=friction, steps=100, dt=0.5, rng=rng)
    return res["speed_hist"]  # steps x N


n_rollouts = 260
param_sets = []
sampler = qmc.LatinHypercube(d=3, seed=7)
lhs = sampler.random(n_rollouts)
T_range = (0.8, 2.2)
a_range = (0.6, 2.0)
fric_range = (0.5, 1.0)
for row in lhs:
    T = T_range[0] + row[0] * (T_range[1] - T_range[0])
    a = a_range[0] + row[1] * (a_range[1] - a_range[0])
    fr = fric_range[0] + row[2] * (fric_range[1] - fric_range[0])
    param_sets.append((T, a, fr))

rollouts = []
for i, (T, a, fr) in enumerate(param_sets):
    hist = make_rollout(seed=2000 + i, T=T, a=a, b=2.0, friction=fr)
    rollouts.append({"hist": hist, "T": T, "a": a, "friction": fr})

# Build supervised dataset: window of mean-speed/std-speed history -> future mean-speed at each horizon
WINDOW = 6


def featurize(hist, t):
    window = hist[t - WINDOW:t]
    mean_v = window.mean(axis=1)
    std_v = window.std(axis=1)
    return np.concatenate([mean_v, std_v])


X_rows, Y_rows = [], []
for r in rollouts:
    hist = r["hist"]
    T_steps = hist.shape[0]
    for t in range(WINDOW, T_steps - max(HORIZONS)):
        feat = featurize(hist, t)
        feat = np.concatenate([feat, [r["T"], r["a"], r["friction"]]])
        target = [hist[t + h].mean() for h in HORIZONS]
        X_rows.append(feat)
        Y_rows.append(target)

X = np.array(X_rows)
Y = np.array(Y_rows)
split = int(0.8 * len(X))
idx_perm = rng_global.permutation(len(X))
X, Y = X[idx_perm], Y[idx_perm]
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

surrogate = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
# multi-output via separate models per horizon (sklearn GBR is single-target)
surrogates = []
preds = np.zeros_like(Y_test)
for h_i in range(len(HORIZONS)):
    m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
    m.fit(X_train, Y_train[:, h_i])
    preds[:, h_i] = m.predict(X_test)
    surrogates.append(m)

baseline_preds = np.tile(X_test[:, WINDOW - 1:WINDOW], (1, len(HORIZONS)))  # naive: hold last mean speed

mape_by_horizon = []
baseline_mape_by_horizon = []
for h_i, h in enumerate(HORIZONS):
    mape = np.mean(np.abs(preds[:, h_i] - Y_test[:, h_i]) / (np.abs(Y_test[:, h_i]) + 1e-3))
    bmape = np.mean(np.abs(baseline_preds[:, h_i] - Y_test[:, h_i]) / (np.abs(Y_test[:, h_i]) + 1e-3))
    mape_by_horizon.append(mape)
    baseline_mape_by_horizon.append(bmape)

# wall-clock speedup: surrogate inference vs simulator for equivalent horizon
n_speed_trials = 30
t_sim0 = time.time()
for _ in range(n_speed_trials):
    make_rollout(seed=int(rng_global.integers(1e6)), T=1.2, a=1.0, b=2.0, friction=0.9)
t_sim_total = time.time() - t_sim0

t_sur0 = time.time()
for _ in range(n_speed_trials):
    surrogates[-1].predict(X_test[:1])
t_sur_total = time.time() - t_sur0
speedup = t_sim_total / max(t_sur_total, 1e-9)

e1_df = pd.DataFrame({
    "horizon_steps": HORIZONS,
    "horizon_seconds": [h * 0.5 for h in HORIZONS],
    "surrogate_MAPE": mape_by_horizon,
    "naive_baseline_MAPE": baseline_mape_by_horizon,
})
e1_df.to_csv(os.path.join(OUT_DIR, "e1_results.csv"), index=False)

plt.figure(figsize=(6, 4))
plt.plot(e1_df["horizon_seconds"], e1_df["surrogate_MAPE"], marker="o", label="GBR surrogate")
plt.plot(e1_df["horizon_seconds"], e1_df["naive_baseline_MAPE"], marker="s", label="naive hold-last baseline")
plt.xlabel("Rollout horizon (s)")
plt.ylabel("Mean speed MAPE")
plt.title("E1: Surrogate multi-step rollout error (WP1 proxy)")
plt.legend()
savefig("fig_e1_surrogate_accuracy.png")

plt.figure(figsize=(4.5, 4))
plt.bar(["Simulator", "Surrogate"], [t_sim_total / n_speed_trials, t_sur_total / n_speed_trials])
plt.ylabel("Wall-clock per rollout query (s)")
plt.title(f"E1: Inference speedup = {speedup:.1f}x")
savefig("fig_e1_speedup.png")

SUMMARY["E1"] = {
    "mape_by_horizon": dict(zip([f"{h*0.5}s" for h in HORIZONS], mape_by_horizon)),
    "baseline_mape_by_horizon": dict(zip([f"{h*0.5}s" for h in HORIZONS], baseline_mape_by_horizon)),
    "speedup_x": float(speedup),
    "n_training_rollouts": n_rollouts,
    "runtime_sec": time.time() - t0,
}
print("E1 speedup:", speedup, "x; MAPE by horizon:", mape_by_horizon)

# ===========================================================================
# E2 -- uncertainty-aware calibration (WP2 SBI-Twin proxy)
# ===========================================================================
print("\n=== E2: uncertainty-aware calibration ===")
t0 = time.time()

def summary_stats(hist):
    mean_v = hist[-40:].mean()
    std_v = hist[-40:].std(axis=1).mean()
    flow_proxy = hist[-40:].mean(axis=1).mean() * N_FIXED
    return [mean_v, std_v, flow_proxy]


def make_calib_dataset(n, friction_regime):
    rows = []
    for i in range(n):
        T = rng_global.uniform(0.8, 2.2)
        a = rng_global.uniform(0.6, 2.0)
        fr = friction_regime
        hist = make_rollout(seed=int(rng_global.integers(1e7)), T=T, a=a, b=2.0, friction=fr)
        stats = summary_stats(hist)
        rows.append({"T": T, "a": a, "friction": fr, **{f"s{j}": s for j, s in enumerate(stats)}})
    return pd.DataFrame(rows)


clear_df = make_calib_dataset(180, friction_regime=1.0)
snow_df = make_calib_dataset(80, friction_regime=0.55)

full_df = pd.concat([clear_df, snow_df], ignore_index=True)
feat_cols = ["s0", "s1", "s2", "friction"]
split2 = int(0.75 * len(full_df))
perm2 = rng_global.permutation(len(full_df))
full_df = full_df.iloc[perm2].reset_index(drop=True)
train_df, test_df = full_df.iloc[:split2], full_df.iloc[split2:]

# Point-estimate baseline: simple linear regression (proxy for grid-search/GP point calibration)
lin_T = LinearRegression().fit(train_df[feat_cols], train_df["T"])
lin_a = LinearRegression().fit(train_df[feat_cols], train_df["a"])
pred_T_lin = lin_T.predict(test_df[feat_cols])
pred_a_lin = lin_a.predict(test_df[feat_cols])

# Uncertainty-aware ensemble ("NPE-mini"): bootstrap ensemble of small regressors -> mean + std
n_ensemble = 12
ens_T_preds, ens_a_preds = [], []
for m_i in range(n_ensemble):
    boot_idx = rng_global.integers(0, len(train_df), size=len(train_df))
    boot = train_df.iloc[boot_idx]
    gbr_T = GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=m_i)
    gbr_a = GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=m_i + 100)
    gbr_T.fit(boot[feat_cols], boot["T"])
    gbr_a.fit(boot[feat_cols], boot["a"])
    ens_T_preds.append(gbr_T.predict(test_df[feat_cols]))
    ens_a_preds.append(gbr_a.predict(test_df[feat_cols]))

ens_T_preds = np.array(ens_T_preds)  # ensemble x n_test
ens_a_preds = np.array(ens_a_preds)
mean_T_ens, std_T_ens = ens_T_preds.mean(axis=0), ens_T_preds.std(axis=0) + 1e-6
mean_a_ens, std_a_ens = ens_a_preds.mean(axis=0), ens_a_preds.std(axis=0) + 1e-6

rmse_lin_T = np.sqrt(np.mean((pred_T_lin - test_df["T"].values) ** 2))
rmse_ens_T = np.sqrt(np.mean((mean_T_ens - test_df["T"].values) ** 2))
rmse_lin_a = np.sqrt(np.mean((pred_a_lin - test_df["a"].values) ** 2))
rmse_ens_a = np.sqrt(np.mean((mean_a_ens - test_df["a"].values) ** 2))


def coverage(mean_pred, std_pred, truth, z):
    lo, hi = mean_pred - z * std_pred, mean_pred + z * std_pred
    return float(np.mean((truth >= lo) & (truth <= hi)))

cov80_T = coverage(mean_T_ens, std_T_ens, test_df["T"].values, 1.2816)
cov95_T = coverage(mean_T_ens, std_T_ens, test_df["T"].values, 1.96)

# per-regime held-out RMSE (clear vs snow) for the ensemble model
snow_mask_test = (test_df["friction"] < 0.9).values
rmse_ens_T_clear = np.sqrt(np.mean((mean_T_ens[~snow_mask_test] - test_df["T"].values[~snow_mask_test]) ** 2))
rmse_ens_T_snow = np.sqrt(np.mean((mean_T_ens[snow_mask_test] - test_df["T"].values[snow_mask_test]) ** 2)) if snow_mask_test.sum() > 0 else float("nan")
rmse_lin_T_snow = np.sqrt(np.mean((pred_T_lin[snow_mask_test] - test_df["T"].values[snow_mask_test]) ** 2)) if snow_mask_test.sum() > 0 else float("nan")

improvement_pct = 100 * (rmse_lin_T - rmse_ens_T) / rmse_lin_T

e2_df = pd.DataFrame([{
    "param": "T (desired headway)",
    "rmse_point_baseline": rmse_lin_T,
    "rmse_ensemble_calibration": rmse_ens_T,
    "rmse_point_baseline_snow": rmse_lin_T_snow,
    "rmse_ensemble_snow": rmse_ens_T_snow,
    "coverage_80pct_nominal": cov80_T,
    "coverage_95pct_nominal": cov95_T,
    "pct_improvement_vs_point": improvement_pct,
}, {
    "param": "a (max accel)",
    "rmse_point_baseline": rmse_lin_a,
    "rmse_ensemble_calibration": rmse_ens_a,
    "rmse_point_baseline_snow": float("nan"),
    "rmse_ensemble_snow": float("nan"),
    "coverage_80pct_nominal": float("nan"),
    "coverage_95pct_nominal": float("nan"),
    "pct_improvement_vs_point": 100 * (rmse_lin_a - rmse_ens_a) / rmse_lin_a,
}])
e2_df.to_csv(os.path.join(OUT_DIR, "e2_results.csv"), index=False)

plt.figure(figsize=(5.5, 4))
order = np.argsort(test_df["T"].values)
plt.errorbar(np.arange(len(order)), mean_T_ens[order], yerr=1.96 * std_T_ens[order], fmt="o", ms=3, alpha=0.6, label="ensemble mean +/- 95%")
plt.plot(np.arange(len(order)), test_df["T"].values[order], "k-", lw=1, label="true T")
plt.xlabel("Held-out test sample (sorted by true T)")
plt.ylabel("Desired headway T (s)")
plt.title("E2: Calibration posterior coverage (WP2 proxy)")
plt.legend()
savefig("fig_e2_calibration_coverage.png")

plt.figure(figsize=(5, 4))
labels = ["Point baseline\n(clear)", "Ensemble\n(clear)", "Point baseline\n(snow)", "Ensemble\n(snow)"]
rmse_lin_T_clear = np.sqrt(np.mean((pred_T_lin[~snow_mask_test] - test_df["T"].values[~snow_mask_test]) ** 2))
vals = [rmse_lin_T_clear, rmse_ens_T_clear, rmse_lin_T_snow, rmse_ens_T_snow]
plt.bar(labels, vals, color=["gray", "steelblue", "gray", "steelblue"])
plt.ylabel("Held-out RMSE on T")
plt.title("E2: Reality gap by regime (WP2 proxy)")
savefig("fig_e2_reality_gap.png")

SUMMARY["E2"] = {
    "rmse_point_T": float(rmse_lin_T),
    "rmse_ensemble_T": float(rmse_ens_T),
    "pct_improvement_T": float(improvement_pct),
    "coverage_80": float(cov80_T),
    "coverage_95": float(cov95_T),
    "runtime_sec": time.time() - t0,
}
print("E2 improvement % vs point baseline:", improvement_pct, "coverage80:", cov80_T, "coverage95:", cov95_T)

# ===========================================================================
# E3 -- belief-state robustness under observation dropout (WP3 proxy)
# ===========================================================================
print("\n=== E3: belief-state robustness under dropout ===")
t0 = time.time()

# Scenario: single-follower car-following pair. The lead vehicle executes a sudden hard
# braking event while the ego (CAV) vehicle's V2X link to the lead experiences bursty dropout.
# This isolates exactly the failure chain both memos describe (missing/stale observation ->
# incorrect local belief -> unsafe action) in a controlled, literature-standard car-following
# setting, rather than the noisier multi-vehicle ring-road wave.


def simulate_platoon_brake_event(mode, dropout_p, rng, brake_start=30, brake_decel=3.5,
                                  brake_duration=16, v0=15.0, gap0=12.0, steps=100, dt=0.5,
                                  burst_mean=6):
    v_lead_true = np.zeros(steps)
    v_lead_true[0] = v0
    for t in range(1, steps):
        if brake_start <= t < brake_start + brake_duration:
            v_lead_true[t] = max(0.0, v_lead_true[t - 1] - brake_decel * dt)
        else:
            v_lead_true[t] = min(v0, v_lead_true[t - 1] + 1.0 * dt)
    x_lead = np.cumsum(v_lead_true) * dt
    x_ego, v_ego = x_lead[0] - gap0, v0
    min_gap = gap0
    state = {"last_gap": gap0, "last_vlead": v0, "steps_since_update": 0,
             "in_burst": False, "burst_remaining": 0}

    for t in range(steps):
        true_gap = x_lead[t] - x_ego
        true_vlead = v_lead_true[t]
        if mode == "privileged":
            use_gap, use_vlead, staleness = true_gap, true_vlead, 0.0
        else:
            if not state["in_burst"] and rng.random() < dropout_p / burst_mean:
                state["in_burst"] = True
                state["burst_remaining"] = max(1, int(rng.exponential(burst_mean)))
            if state["in_burst"]:
                use_gap, use_vlead = state["last_gap"], state["last_vlead"]
                state["steps_since_update"] += 1
                state["burst_remaining"] -= 1
                if state["burst_remaining"] <= 0:
                    state["in_burst"] = False
            else:
                use_gap, use_vlead = true_gap, true_vlead
                state["last_gap"], state["last_vlead"] = true_gap, true_vlead
                state["steps_since_update"] = 0
            staleness = min(state["steps_since_update"] / 8.0, 1.0)

        base_safe_gap = 2.0 + v_ego * 1.3
        # belief mode: inflate the safety margin as confidence in the last V2X update decays
        safe_gap = base_safe_gap * (1.0 + 1.5 * staleness) if mode == "belief" else base_safe_gap
        rel_v = v_ego - use_vlead
        accel = 0.3 * (use_vlead - v_ego)
        if use_gap < safe_gap:
            accel = min(accel, -0.9 * max(0.0, safe_gap - use_gap) - 0.3 * max(0.0, rel_v))
        accel = float(np.clip(accel, -4.0, 1.5))
        v_ego = max(0.0, v_ego + accel * dt)
        x_ego += v_ego * dt
        min_gap = min(min_gap, true_gap)
    return min_gap


dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
modes = ["privileged", "naive", "belief"]
n_trials_e3 = 60
e3_rows = []
for mode in modes:
    for dp in dropouts:
        rng_c = np.random.default_rng(5000 + hash((mode, dp)) % 1000)
        min_gaps = [
            simulate_platoon_brake_event(mode, dp if mode != "privileged" else 0.0, rng_c)
            for _ in range(n_trials_e3)
        ]
        min_gaps = np.array(min_gaps)
        collision_rate = float(np.mean(min_gaps <= 0))
        near_miss_rate = float(np.mean((min_gaps > 0) & (min_gaps < 2.0)))
        e3_rows.append({
            "mode": mode, "dropout": dp,
            "min_gap_mean": float(np.mean(min_gaps)),
            "min_gap_p10": float(np.percentile(min_gaps, 10)),
            "collision_rate": collision_rate,
            "near_miss_rate_TTC_under_2s": near_miss_rate,
        })

e3_df = pd.DataFrame(e3_rows)
e3_df.to_csv(os.path.join(OUT_DIR, "e3_results.csv"), index=False)

plt.figure(figsize=(6, 4))
for mode in modes:
    sub = e3_df[e3_df["mode"] == mode]
    plt.plot(sub["dropout"], sub["collision_rate"], marker="o", label=mode)
plt.xlabel("V2X observation dropout probability")
plt.ylabel("Collision rate (min true gap <= 0) over hard-brake trials")
plt.title("E3: Safety under dropout during lead hard-brake (WP3 proxy)")
plt.legend()
savefig("fig_e3_degradation_curves.png")

# graceful degradation slope: linear fit of collision rate vs dropout, per mode (lower slope = more graceful)
slopes = {}
for mode in modes:
    sub = e3_df[e3_df["mode"] == mode]
    if sub["dropout"].nunique() > 1:
        slope = np.polyfit(sub["dropout"], sub["collision_rate"], 1)[0]
        slopes[mode] = float(slope)

SUMMARY["E3"] = {
    "collision_rate_slope_per_mode": slopes,
    "collision_rate_at_50pct_dropout": e3_df[(e3_df["dropout"] == 0.5)][["mode", "collision_rate"]].to_dict("records"),
    "collision_rate_at_30pct_dropout": e3_df[(e3_df["dropout"] == 0.3)][["mode", "collision_rate"]].to_dict("records"),
    "runtime_sec": time.time() - t0,
}
print("E3 collision-rate slopes vs dropout (lower = more graceful):", slopes)

# ===========================================================================
# E4 -- early wave-onset prediction / transfer (WP4 proxy)
# ===========================================================================
print("\n=== E4: early cascade/wave-onset prediction ===")
t0 = time.time()

n_e4 = 260
e4_records = []
for i in range(n_e4):
    N = int(rng_global.integers(10, 55))
    T = rng_global.uniform(0.8, 2.2)
    a = rng_global.uniform(0.6, 2.0)
    fr = rng_global.uniform(0.5, 1.0)
    res = simulate_ring(N=N, L=L, v0=V0, T=T, a=a, b=2.0, s0=2.0, friction=fr,
                         steps=120, dt=0.5, rng=np.random.default_rng(9000 + i))
    early_hist = res["speed_hist"][:30]
    early_mean = early_hist.mean(axis=1)
    early_std = early_hist.std(axis=1)
    label = int(res["std_speed"][-30:].mean() > 0.15)
    e4_records.append({
        "N": N, "density": N / L, "T": T, "a": a, "friction": fr,
        "early_mean_speed_mean": early_mean.mean(),
        "early_mean_speed_trend": np.polyfit(np.arange(len(early_mean)), early_mean, 1)[0],
        "early_std_mean": early_std.mean(),
        "early_std_trend": np.polyfit(np.arange(len(early_std)), early_std, 1)[0],
        "wave_label": label,
    })

e4_df = pd.DataFrame(e4_records)

# Held-out transfer test: hold out a density band [0.10, 0.16) entirely from training
transfer_band = (e4_df["density"] >= 0.10) & (e4_df["density"] < 0.16)
train4 = e4_df[~transfer_band].reset_index(drop=True)
transfer4 = e4_df[transfer_band].reset_index(drop=True)

feat4 = ["early_mean_speed_mean", "early_mean_speed_trend", "early_std_mean", "early_std_trend", "density"]
split4 = int(0.75 * len(train4))
perm4 = rng_global.permutation(len(train4))
train4 = train4.iloc[perm4].reset_index(drop=True)
tr4, te4 = train4.iloc[:split4], train4.iloc[split4:]

clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
clf.fit(tr4[feat4], tr4["wave_label"])
probs_test = clf.predict_proba(te4[feat4])[:, 1]

if te4["wave_label"].nunique() > 1:
    auroc = roc_auc_score(te4["wave_label"], probs_test)
    auprc = average_precision_score(te4["wave_label"], probs_test)
else:
    auroc, auprc = float("nan"), float("nan")

if len(transfer4) > 0 and transfer4["wave_label"].nunique() > 1:
    probs_transfer = clf.predict_proba(transfer4[feat4])[:, 1]
    auroc_transfer = roc_auc_score(transfer4["wave_label"], probs_transfer)
else:
    auroc_transfer = float("nan")

transfer_degradation_pct = (
    100 * (auroc - auroc_transfer) / auroc if not np.isnan(auroc) and not np.isnan(auroc_transfer) and auroc > 0 else float("nan")
)

e4_df.to_csv(os.path.join(OUT_DIR, "e4_results.csv"), index=False)

plt.figure(figsize=(5.5, 4))
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(te4["wave_label"], probs_test)
plt.plot(fpr, tpr, label=f"held-out (AUROC={auroc:.2f})")
if len(transfer4) > 0 and transfer4["wave_label"].nunique() > 1:
    fpr_t, tpr_t, _ = roc_curve(transfer4["wave_label"], probs_transfer)
    plt.plot(fpr_t, tpr_t, label=f"density-band transfer (AUROC={auroc_transfer:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.7)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("E4: Early wave-onset prediction (WP4 proxy)")
plt.legend()
savefig("fig_e4_onset_auroc.png")

plt.figure(figsize=(5, 4))
plt.bar(["In-distribution\nheld-out", "Held-out density\nband (transfer)"], [auroc, auroc_transfer])
plt.ylim(0, 1)
plt.ylabel("AUROC")
plt.title("E4: Zero-shot transfer degradation (WP4 proxy)")
savefig("fig_e4_transfer.png")

SUMMARY["E4"] = {
    "auroc_heldout": float(auroc),
    "auprc_heldout": float(auprc),
    "auroc_density_transfer": float(auroc_transfer),
    "transfer_degradation_pct": float(transfer_degradation_pct),
    "runtime_sec": time.time() - t0,
}
print("E4 AUROC held-out:", auroc, "transfer:", auroc_transfer)

# ===========================================================================
# E5 -- guided vs random scenario discovery (WP5 proxy)
# ===========================================================================
print("\n=== E5: guided vs random scenario discovery ===")
t0 = time.time()

param_space_low = np.array([10, 0.8, 0.6, 0.5])   # N, T, a, friction
param_space_high = np.array([55, 2.2, 2.0, 1.0])


def sim_label(params):
    N, T, a, fr = params
    N = int(round(N))
    res = simulate_ring(N=N, L=L, v0=V0, T=T, a=a, b=2.0, s0=2.0, friction=fr,
                         steps=110, dt=0.5, rng=np.random.default_rng(int(abs(hash((N, T, a, fr))) % (2**31))))
    return int(res["std_speed"][-30:].mean() > 0.15)


BUDGET = 90
BATCH = 15

# (a) Latin-hypercube / random baseline
sampler5 = qmc.LatinHypercube(d=4, seed=123)
lhs5 = sampler5.random(BUDGET)
random_points = param_space_low + lhs5 * (param_space_high - param_space_low)
random_labels = [sim_label(p) for p in random_points]
random_discoveries = int(np.sum(random_labels))

# (b) guided sampler: start with a small LHS seed batch, then bias sampling using the E4 classifier's
# analogous local density-based heuristic (near current best-known critical band) plus exploration noise
seed_batch_n = 20
sampler5b = qmc.LatinHypercube(d=4, seed=321)
seed_points = param_space_low + sampler5b.random(seed_batch_n) * (param_space_high - param_space_low)
seed_labels = [sim_label(p) for p in seed_points]

guided_points = list(seed_points)
guided_labels = list(seed_labels)

remaining_budget = BUDGET - seed_batch_n
rng_guide = np.random.default_rng(555)
while remaining_budget > 0:
    batch_n = min(BATCH, remaining_budget)
    pts_arr = np.array(guided_points)
    lbls_arr = np.array(guided_labels)
    if lbls_arr.sum() > 0:
        positive_pts = pts_arr[lbls_arr == 1]
        centers = positive_pts[rng_guide.integers(0, len(positive_pts), size=batch_n)]
        noise = rng_guide.normal(0, 1, size=(batch_n, 4)) * (param_space_high - param_space_low) * 0.08
        new_pts = np.clip(centers + noise, param_space_low, param_space_high)
    else:
        new_pts = param_space_low + qmc.LatinHypercube(d=4, seed=int(rng_guide.integers(1e6))).random(batch_n) * (param_space_high - param_space_low)
    new_labels = [sim_label(p) for p in new_pts]
    guided_points.extend(list(new_pts))
    guided_labels.extend(new_labels)
    remaining_budget -= batch_n

guided_discoveries = int(np.sum(guided_labels))

# cumulative discovery curves for plotting
def cumulative_discoveries(labels):
    return np.cumsum(labels)

cum_random = cumulative_discoveries(random_labels)
cum_guided = cumulative_discoveries(guided_labels[:BUDGET])

e5_df = pd.DataFrame({
    "n_simulations": np.arange(1, BUDGET + 1),
    "cumulative_discoveries_random": cum_random,
    "cumulative_discoveries_guided": cum_guided,
})
e5_df.to_csv(os.path.join(OUT_DIR, "e5_results.csv"), index=False)

plt.figure(figsize=(5.5, 4))
plt.plot(e5_df["n_simulations"], e5_df["cumulative_discoveries_random"], label="Random / LHS")
plt.plot(e5_df["n_simulations"], e5_df["cumulative_discoveries_guided"], label="Guided (label-biased) sampler")
plt.xlabel("Number of full simulator runs")
plt.ylabel("Cumulative wave-onset scenarios discovered")
plt.title("E5: Scenario discovery efficiency (WP5 proxy)")
plt.legend()
savefig("fig_e5_discovery_efficiency.png")

discovery_ratio = guided_discoveries / max(random_discoveries, 1)

SUMMARY["E5"] = {
    "budget": BUDGET,
    "random_discoveries": random_discoveries,
    "guided_discoveries": guided_discoveries,
    "discovery_ratio_guided_over_random": float(discovery_ratio),
    "runtime_sec": time.time() - t0,
}
print("E5 discoveries -- random:", random_discoveries, "guided:", guided_discoveries, "ratio:", discovery_ratio)

# ===========================================================================
# wrap up
# ===========================================================================
SUMMARY["total_runtime_sec"] = time.time() - t_global_start
with open(os.path.join(OUT_DIR, "summary_metrics.json"), "w") as f:
    json.dump(SUMMARY, f, indent=2)

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE. Total runtime (s):", SUMMARY["total_runtime_sec"])
print(json.dumps(SUMMARY, indent=2))
