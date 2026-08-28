"""E1-real -- Multi-step speed-trajectory surrogate on REAL NGSIM car-following
episodes (real-data analogue of pilot E1's learned-surrogate experiment).

Design.
  * Unit of prediction: an anchor point (episode, t) with a trailing 2 s
    window of state (v_foll, v_lead, dv, s) as input.
  * Target: the follower's speed at t+1s, t+2s, t+4s, t+8s.
  * Learned surrogate: gradient-boosted regressor per horizon, trained on
    window summary features.
  * Baselines:
      - persistence: v_foll stays at its current value
      - constant-acceleration: extrapolate using the current smoothed accel
      - IDM-physics: integrate the calibrated population-median IDM
        (T, a, b=2, s0=2) forward from the anchor state using the actual
        realized leader trajectory (mirrors the pilot's "run the simulator"
        comparison -- here the "simulator" is the physics model, and it gets
        the same leader-trajectory information the follower actually saw).
  * Split: GROUPED BY VEHICLE ID (same leakage fix as E2-real) -- windows
    from one vehicle's episode never appear on both sides.

Outputs: e1_real_results.csv, fig_e1_real_accuracy.png, e1_real_summary.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

import ngsim_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'experiment_results')
CACHE = os.path.join(HERE, 'cache')
os.makedirs(OUT, exist_ok=True)

WIN_S = 2.0          # trailing input window
HORIZONS_S = [1.0, 2.0, 4.0, 8.0]
STRIDE_S = 1.0        # spacing between sampled anchors within an episode
FPS = 10.0
T_MEDIAN, A_MEDIAN = 1.031, 0.458  # population median of e2_real's causal fits
B_FIXED, S0_FIXED = 2.0, 2.0


def build_windows(eps):
    win = int(WIN_S * FPS)
    hors = [int(h * FPS) for h in HORIZONS_S]
    stride = int(STRIDE_S * FPS)
    max_h = max(hors)
    rows = []
    for eid, e in eps.groupby('episode_id'):
        e = e.sort_values('global_time')
        n = len(e)
        if n < win + max_h + 1:
            continue
        v = e.v_foll.values
        vl = e.v_lead.values
        dv = e.dv.values
        s = e.s.values
        vid = int(e.vehicle_id.iloc[0])
        for t in range(win, n - max_h, stride):
            w = slice(t - win, t)
            row = {
                'episode_id': eid, 'vehicle_id': vid, 't_idx': t,
                'v_now': v[t - 1], 's_now': s[t - 1], 'dv_now': dv[t - 1],
                'v_mean': v[w].mean(), 'v_std': v[w].std(),
                'v_slope': (v[t - 1] - v[t - win]) / WIN_S,
                's_mean': s[w].mean(), 'dv_mean': dv[w].mean(),
                'vl_now': vl[t - 1],
            }
            for h, hs in zip(hors, HORIZONS_S):
                row[f'y_{hs}'] = v[t - 1 + h]
                row[f'vl_path_{hs}'] = vl[t - 1:t - 1 + h + 1].tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def idm_rollout(v0, s0, dv0, vl_path, T=T_MEDIAN, a=A_MEDIAN,
               b=B_FIXED, s0_jam=S0_FIXED, dt=0.1, v0_freeflow=29.06):
    """Integrate IDM forward one follower, GIVEN the real leader speed path
    it actually observed (removes leader-prediction error from this test --
    isolates follower-response quality, the part E1 is about)."""
    v, s = v0, s0
    for k in range(1, len(vl_path)):
        s_star = s0_jam + max(v * T + v * (v - vl_path[k - 1]) /
                              (2 * np.sqrt(a * b)), 0.0)
        acc = a * (1 - (v / v0_freeflow) ** 4 - (s_star / max(s, 0.1)) ** 2)
        acc = np.clip(acc, -9.0, 4.0)
        v_new = max(v + acc * dt, 0.0)
        s = s + (vl_path[k - 1] - v) * dt
        v = v_new
    return v


FEATS = ['v_now', 's_now', 'dv_now', 'v_mean', 'v_std', 'v_slope',
         's_mean', 'dv_mean']


def main():
    eps = pd.read_parquet(os.path.join(CACHE, 'episodes.parquet'))
    W = build_windows(eps)
    print(f'windows: {len(W)}  episodes: {W.episode_id.nunique()}  '
         f'vehicles: {W.vehicle_id.nunique()}')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    tr_idx, te_idx = next(gss.split(W, groups=W.vehicle_id.values))
    Wtr, Wte = W.iloc[tr_idx], W.iloc[te_idx]
    assert not (set(Wtr.vehicle_id) & set(Wte.vehicle_id))

    Xtr = Wtr[FEATS].values
    results = []
    for hs in HORIZONS_S:
        ytr = Wtr[f'y_{hs}'].values
        yte = Wte[f'y_{hs}'].values
        Xte = Wte[FEATS].values

        gb = GradientBoostingRegressor(random_state=0, max_depth=3,
                                       n_estimators=200)
        gb.fit(Xtr, ytr)
        pred_gb = gb.predict(Xte)

        pred_persist = Wte['v_now'].values
        pred_ca = np.clip(Wte['v_now'].values + Wte['v_slope'].values * hs,
                          0.0, None)
        pred_idm = np.array([
            idm_rollout(r.v_now, r.s_now, r.dv_now, r.vl_path)
            for r in Wte[['v_now', 's_now', 'dv_now', f'vl_path_{hs}']]
                        .rename(columns={f'vl_path_{hs}': 'vl_path'})
                        .itertuples()
        ])

        eps_ = 1e-6
        def mape(pred):
            return float(np.mean(np.abs(pred - yte) / np.maximum(yte, 1.0)) * 100)

        results.append({
            'horizon_s': hs,
            'mape_gb': mape(pred_gb),
            'mape_persistence': mape(pred_persist),
            'mape_const_accel': mape(pred_ca),
            'mape_idm_physics': mape(pred_idm),
            'rmse_gb': float(np.sqrt(np.mean((pred_gb - yte) ** 2))),
            'rmse_persistence': float(np.sqrt(np.mean((pred_persist - yte) ** 2))),
            'rmse_idm_physics': float(np.sqrt(np.mean((pred_idm - yte) ** 2))),
        })

    R = pd.DataFrame(results)
    R.to_csv(os.path.join(OUT, 'e1_real_results.csv'), index=False)
    json.dump({'n_windows': int(len(W)), 'n_train': int(len(Wtr)),
               'n_test': int(len(Wte)),
               'n_train_vehicles': int(Wtr.vehicle_id.nunique()),
               'n_test_vehicles': int(Wte.vehicle_id.nunique()),
               'idm_population_params': {'T': T_MEDIAN, 'a': A_MEDIAN},
               'results': results},
              open(os.path.join(OUT, 'e1_real_summary.json'), 'w'), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for col, lab, mk in [('mape_gb', 'GB surrogate', 'o-'),
                         ('mape_persistence', 'persistence', 's--'),
                         ('mape_const_accel', 'const-accel extrap.', '^--'),
                         ('mape_idm_physics', 'IDM physics (pop. params)', 'd--')]:
        ax.plot(R.horizon_s, R[col], mk, label=lab)
    ax.set_xlabel('prediction horizon [s]'); ax.set_ylabel('MAPE [%]')
    ax.set_title('E1-real: multi-step speed prediction, real NGSIM')
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.2
    x = np.arange(len(R))
    for i, (col, lab) in enumerate([('rmse_gb', 'GB'), ('rmse_persistence', 'persist'),
                                    ('rmse_idm_physics', 'IDM physics')]):
        ax.bar(x + (i - 1) * width, R[col], width=width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f'{h:g}s' for h in R.horizon_s])
    ax.set_xlabel('horizon'); ax.set_ylabel('RMSE [m/s]')
    ax.set_title('RMSE by horizon'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_e1_real_accuracy.png'), dpi=140)
    print(R.to_string(index=False))


if __name__ == '__main__':
    main()
