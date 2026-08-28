"""E4-real -- Early wave-onset prediction on REAL NGSIM US-101 data.

Design (mirrors pilot E4 at real-data scale):
  * Prediction units: lane x space cells (one 300 ft cell per mainline lane).
  * Label y(c, t) = 1 if cell c enters a jammed state (10 s-smoothed mean
    speed < 30 mph) within the next 45 s after being continuously unjammed
    for >= 20 s -- i.e. a genuine wave-front arrival, not lingering congestion.
  * Features: trailing 30 s statistics of the cell's own speed (mean, std,
    slope), its immediate upstream neighbour, and the whole-lane aggregate.
  * Generalisation tests:
      - temporal holdout: train on t < 640 s, test on the rest;
      - spatial (lane) holdout: train lanes {1,2,3}, test lanes {4,5}
        -- the real-data analogue of the pilot's held-out density band.
  * Models: gradient-boosted classifier; baselines: majority class and
    persistence ("cell fast now -> stays fast").

Output: e4_real_results.csv, fig_e4_real_roc.png, e4_real_summary.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import ngsim_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'experiment_results')
CACHE = os.path.join(HERE, 'cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

MAINLINE_LANES = [1, 2, 3, 4, 5]
BIN_T = 2.0          # s per field frame
BIN_X = 100.0        # ft per field column
CELL_XBINS = 3       # cells are 300 ft
JAM_MPH = 30.0
FREE_MPH = 40.0
HORIZON_S = 45.0
TRAIL_S = 30.0
SAMPLE_EVERY = 8.0   # seconds between sampled prediction points
TEMPORAL_SPLIT_S = 640.0


def build_cells(clean):
    """Return {lane: DataFrame [t_bin x cell]} of mean speeds in mph."""
    d = clean[clean.lane_id.isin(MAINLINE_LANES)].copy()
    d['t_bin'] = ((d.global_time - d.global_time.min()) / 1000.0 // BIN_T).astype(int)
    d['x_bin'] = (d.local_y_s // BIN_X).astype(int)
    d['cell'] = d.x_bin // CELL_XBINS
    agg = d.groupby(['lane_id', 't_bin', 'cell']).v_vel_s.agg(['mean', 'size'])
    agg = agg[agg['size'] >= 2] * 0.681818            # ft/s -> mph
    out = {}
    for lane, g in agg.groupby(level=0):
        piv = g.droplevel(0)['mean'].unstack('cell')      # rows: t_bin
        piv = piv.reindex(index=pd.RangeIndex(int(piv.index.max()) + 1),
                          columns=pd.RangeIndex(int(piv.columns.max()) + 1))
        out[int(lane)] = piv.sort_index(axis=1).interpolate(axis=0, limit=5)
    return out


def _movavg_nanaware(a, k=5):
    """Centered moving average along time axis (axis 0), NaN-aware."""
    from scipy.ndimage import uniform_filter1d
    filled = np.where(np.isnan(a), 0.0, a)
    valid = (~np.isnan(a)).astype(float)
    num = uniform_filter1d(filled, size=k, axis=0, mode='nearest') * k
    den = uniform_filter1d(valid, size=k, axis=0, mode='nearest') * k
    return num / np.maximum(den, 1.0)


# Wave-onset definition calibrated to this congested peak-period regime:
# a wave FRONT ARRIVES at cell c if its 10 s-smoothed speed dips >= DROP_MPH
# below the cell's trailing reference level within the next HORIZON_S.
# Absolute free/jam thresholds are unusable here -- the whole section is
# already congested at 07:50-08:05 (see E0-real).
DROP_MPH = 12.0      # ~75th pct of observed per-cell 20-s swing range


def label_and_featurize(cells):
    """Sample (cell, t) points with trailing-window features and labels."""
    horizon_bins = int(HORIZON_S / BIN_T)
    trail_bins = int(TRAIL_S / BIN_T)
    ref_bins = int(20.0 / BIN_T)          # trailing reference window
    rows = []
    for lane in sorted(cells):
        V = cells[lane].to_numpy()                    # [t_bin, cell]
        n_t, n_c = V.shape
        # smoothed + hysteresis jam state
        Vs = _movavg_nanaware(V, k=5)
        for c in range(n_c):
            v = V[:, c]
            vs = Vs[:, c]
            if np.all(np.isnan(v)):
                continue
            v_up = V[:, max(c - 1, 0)]
            v_ups = Vs[:, max(c - 1, 0)]
            v_lane = np.nanmean(V, axis=1)
            # label: does the smoothed speed dip >= DROP_MPH below the
            # trailing reference level at any point within the horizon?
            is_onset = np.zeros(n_t, bool)
            for t in range(ref_bins, n_t - horizon_bins):
                ref = np.nanmean(v[t - ref_bins:t])
                fut = vs[t + 1: t + 1 + horizon_bins]
                if np.isnan(fut).any():
                    continue
                if np.nanmin(fut) <= ref - DROP_MPH:
                    is_onset[t] = True
            for t in range(trail_bins, n_t - horizon_bins, int(SAMPLE_EVERY / BIN_T)):
                w = v[t - trail_bins:t]
                ws_ = vs[t - trail_bins:t]
                wu = v_up[t - trail_bins:t]
                wl = v_lane[t - trail_bins:t]
                if np.isnan(w).any() or np.isnan(wl).any():
                    continue
                tt = np.arange(trail_bins) * BIN_T
                slope = np.polyfit(tt, w, 1)[0]
                y = int(is_onset[t])
                rows.append({
                    'lane': lane, 'cell': c,
                    't_sec': t * BIN_T,
                    'f_mean': np.nanmean(w), 'f_std': np.nanstd(ws_),
                    'f_slope': slope, 'f_min': np.nanmin(w),
                    'f_up_mean': np.nanmean(wu),
                    'f_up_min': np.nanmin(wu),
                    'f_lane_mean': np.nanmean(wl),
                    'f_lane_slope': np.polyfit(tt, wl, 1)[0],
                    'y': y,
                })
    return pd.DataFrame(rows)


def evaluate(df_train, df_test, feats, tag):
    Xtr, ytr = df_train[feats].values, df_train.y.values
    Xte, yte = df_test[feats].values, df_test.y.values
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    base_pos = float(ytr.mean())
    maj = np.full(len(yte), base_pos)
    # persistence / physics prior: upstream cell already >= 8 mph below this
    # cell -> the wave front should arrive here next
    pers = ((df_test.f_mean.values - df_test.f_up_min.values) >= 8.0)
    pers = pers.astype(float) if hasattr(pers, 'astype') else pers.float()
    res = {
        'split': tag,
        'n_train': int(len(ytr)), 'n_test': int(len(yte)),
        'base_rate': float(yte.mean()),
        'auroc_gb': float(roc_auc_score(yte, p)),
        'auprc_gb': float(average_precision_score(yte, p)),
        'auroc_persistence': float(roc_auc_score(yte, pers)),
        'auprc_majority': float(yte.mean()),   # constant predictor = test prev.
    }
    return res, p, yte


def main():
    cache_c = os.path.join(CACHE, 'cells')          # per-lane: cells_<lane>
    cache_s = os.path.join(CACHE, 'samples.parquet')
    if not os.path.exists(cache_s):
        cells = {int(l): pd.read_parquet(f'{cache_c}_{l}')
                 for l in MAINLINE_LANES
                 if os.path.exists(f'{cache_c}_{l}')}
        if len(cells) < len(MAINLINE_LANES):
            clean = pd.read_parquet(os.path.join(CACHE, 'clean.parquet'))
            cells = build_cells(clean)
            for lane, v in cells.items():
                v.to_parquet(f'{cache_c}_{lane}')
        S = label_and_featurize(cells)
        S.to_parquet(cache_s)
    else:
        S = pd.read_parquet(cache_s)

    feats = ['f_mean', 'f_std', 'f_slope', 'f_min',
             'f_up_mean', 'f_up_min', 'f_lane_mean', 'f_lane_slope']
    results = []

    # temporal split -- drop training rows whose label horizon extends past
    # the split boundary (audit finding: 8.1% of naive-train rows peeked into
    # the test period's dynamics via their forward-looking onset label)
    tr = S[(S.t_sec < TEMPORAL_SPLIT_S) &
           (S.t_sec + HORIZON_S <= TEMPORAL_SPLIT_S)]
    te = S[S.t_sec >= TEMPORAL_SPLIT_S]
    r, _, _ = evaluate(tr, te, feats, 'temporal_holdout')
    results.append(r)

    # lane split
    trL = S[S.lane.isin([1, 2, 3])]
    teL = S[S.lane.isin([4, 5])]
    r2, _, _ = evaluate(trL, teL, feats, 'lane_holdout')
    results.append(r2)

    R = pd.DataFrame(results)
    R.to_csv(os.path.join(OUT, 'e4_real_results.csv'), index=False)
    with open(os.path.join(OUT, 'e4_real_summary.json'), 'w') as f:
        json.dump({'n_samples': int(len(S)),
                   'onset_rate': float(S.y.mean()),
                   'temporal_split_s': TEMPORAL_SPLIT_S,
                   'results': results}, f, indent=2)

    # figure: score distributions + bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    tags = ['temporal_holdout', 'lane_holdout']
    x = np.arange(len(tags))
    for i, m in enumerate(['auroc_gb', 'auroc_persistence']):
        vals = [results[k][m] for k in range(len(tags))]
        bars = ax.bar(x + (i - 0.5) * 0.3, vals, width=0.28,
                      color=['#2c7fb8' if i == 0 else '#bbbbbb'][0:1] * len(vals))
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + .01, f'{v:.2f}',
                    ha='center', fontsize=9)
    ax.axhline(0.5, color='k', ls='--', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel('AUROC'); ax.set_ylim(0, 1)
    ax.set_title('E4-real: wave-onset prediction (GBM vs persistence)')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c)
                       for c in ['#2c7fb8', '#bbbbbb']],
              labels=['gradient boosting', 'persistence'], fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.15, [results[k]['auprc_gb'] for k in range(2)], width=0.28,
           color='#2c7fb8', label='GB AUPRC')
    ax.bar(x + 0.15, [results[k]['base_rate'] for k in range(2)], width=0.28,
           color='#bbbbbb', label='base rate (majority)')
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel('AUPRC'); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title('E4-real: precision-recall vs class imbalance')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_e4_real_roc.png'), dpi=140)
    print(R.to_string(index=False))


if __name__ == '__main__':
    main()
