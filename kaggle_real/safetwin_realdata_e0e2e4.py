"""
SafeTwin pilot -- REAL-DATA experiments (E0/E1/E2/E4) on NGSIM US-101.
Self-contained Kaggle kernel script. Data source (attached dataset):
    nigelwilliams/ngsim-vehicle-trajectory-data-us-101
Outputs -> /kaggle/working/

E0-real  sanity gate: space-time speed field shows stop-and-go waves.
E1-real  multi-step follower-speed prediction: learned surrogate vs.
         persistence / constant-accel / IDM-physics-rollout baselines.
E2-real  uncertainty-aware IDM calibration on real car-following episodes;
         bootstrap UQ vs scaled split-conformal intervals. Train/test AND
         the inner bootstrap fit/calibration split are GROUPED BY VEHICLE ID
         (a random episode-level split leaks driver style: 37% of episodes
         share a vehicle with >=1 other episode). Target theta=(T,a) is
         fit on the SECOND HALF of each episode only, strictly disjoint in
         time from the first-half summary features the calibrator sees.
E4-real  wave-onset prediction from early partial windows; temporal and
         lane-holdout generalisation tests. Training rows whose forward
         label horizon crosses the temporal-split boundary are dropped.

Methodology audit (2026-08-23): the vehicle-grouping and causal-target-fit
fixes were added after a leakage audit found 37% episode/vehicle overlap
under the original random split and only 0.73/0.51 correlation between
full-episode vs. second-half-only IDM refits (i.e., the original full-
episode target was not fully independent of the first-half features).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

OUT = '/kaggle/working'


def _find_ngsim():
    """Locate the trajectory file wherever the dataset is mounted."""
    import glob
    root = os.environ.get('NGSIM_ROOT', '/kaggle/input')
    pats = [root + '/**/trajectories-0750am-0805am.txt', root + '/**/*.txt']
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        for h in hits:
            if '0750' in os.path.basename(h):
                return h
        if hits:
            return hits[0]
    raise FileNotFoundError('NGSIM trajectories-0750am-0805am.txt not found '
                            'under /kaggle/input')


NGSIM_TXT = _find_ngsim()

# ------------------------------------------------------------------ config
NGSIM_COLS = ['vehicle_id', 'frame_id', 'total_frames', 'global_time',
              'local_x', 'local_y', 'global_x', 'global_y',
              'v_length', 'v_width', 'v_class', 'v_vel', 'v_acc',
              'lane_id', 'preceding', 'following', 'space_hdwy', 'time_hdwy']
FT2M = 0.3048
V0_DEFAULT = 29.0576          # section free-flow speed, m/s
ACCEL_PLAUS_FT2 = 25.0        # ~0.77 g tracker-swap filter
MAINLINE_LANES = [1, 2, 3, 4, 5]
BIN_T, BIN_X, CELL_XBINS = 2.0, 100.0, 3
DROP_MPH = 12.0
HORIZON_S, TRAIL_S, SAMPLE_EVERY = 45.0, 30.0, 8.0
TEMPORAL_SPLIT_S = 640.0
MIN_FRAMES = 40                # min episode length for E2's causal split fit


def log(msg):
    print(f'[{__import__("time").strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------- load/clean
def load_ngsim(path):
    df = pd.read_csv(path, sep=r'\s+', names=NGSIM_COLS, engine='c')
    return df[df.v_class.isin([2, 3])].copy()


def clean_trajectories(df, window=13, poly=2):
    """Position-first cleaning: S-G smooth local_y per gap-free track
    segment, derive v = dy/dt, a = dv/dt. Flags implausible frames."""
    y_s_all = pd.Series(np.nan, index=df.index)
    v_s_all = pd.Series(np.nan, index=df.index)
    a_s_all = pd.Series(np.nan, index=df.index)
    for vid, g in df.groupby('vehicle_id', sort=False):
        g = g.sort_values('global_time')
        idx = g.index.to_numpy()
        breaks = np.flatnonzero(np.diff(g.frame_id.values) != 1) + 1
        for ix in np.split(np.arange(len(g)), breaks):
            n = len(ix)
            y = g.local_y.values[ix]
            if n >= 7:
                w = min(window if window % 2 else window + 1,
                        n if n % 2 else n - 1)
                y_s = savgol_filter(y, window_length=w, polyorder=poly)
                v_s = np.gradient(y_s, 0.1)
                a_s = np.gradient(v_s, 0.1)
            elif n >= 2:
                y_s = y
                v_s = np.gradient(y, 0.1)
                a_s = np.gradient(v_s, 0.1)
            else:
                y_s, v_s, a_s = y, g.v_vel.values[ix], g.v_acc.values[ix]
            y_s_all[idx] = y_s
            v_s_all[idx] = v_s
            a_s_all[idx] = a_s
    out = df.copy()
    out['local_y_s'] = y_s_all.to_numpy()
    out['v_vel_s'] = v_s_all.to_numpy()
    out['v_acc_s'] = a_s_all.to_numpy()
    ok = out.v_vel_s.notna()
    out['bad_acc'] = (ok & (out.v_acc_s.abs() > ACCEL_PLAUS_FT2)).fillna(False)
    return out


# ------------------------------------------------------------------ E0
def e0_real(clean):
    d = clean[clean.lane_id.isin(MAINLINE_LANES)].copy()
    t0g = d.global_time.min()
    d['t_bin'] = ((d.global_time - t0g) / 1000.0 // BIN_T).astype(int)
    d['x_bin'] = (d.local_y_s // BIN_X).astype(int)
    fields = {}
    for lane, g in d.groupby('lane_id'):
        agg = g.groupby(['t_bin', 'x_bin']).v_vel_s.agg(['mean', 'size'])
        agg = agg[agg['size'] >= 2]
        piv = agg['mean'].unstack('x_bin') * 0.681818
        fields[int(lane)] = piv
    lanes = sorted(fields)
    t_bins = max(f.index.max() for f in fields.values()) + 1
    x_bins = max(f.columns.max() for f in fields.values()) + 1
    grid = np.full((len(lanes), t_bins, x_bins), np.nan)
    for k, lane in enumerate(lanes):
        f = fields[lane]
        grid[k][np.ix_(f.index.to_numpy(), f.columns.to_numpy())] = f.to_numpy()

    fig, axes = plt.subplots(len(lanes), 1, figsize=(11, 2.1 * len(lanes)),
                             sharex=True, sharey=True)
    extent = [0, x_bins * BIN_X, t_bins * BIN_T, 0]
    for k, ax in enumerate(axes):
        im = ax.imshow(grid[k], aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=70, extent=extent)
        ax.set_ylabel(f'lane {lanes[k]}\ntime [s]')
        plt.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel('space along US-101 [ft]')
    fig.suptitle('E0-real: NGSIM US-101 space-time speed field (07:50-08:05)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_e0_real_speedfield.png', dpi=140)

    per_lane = {}
    for k, lane in enumerate(lanes):
        valid = grid[k][~np.isnan(grid[k])]
        per_lane[str(lane)] = {
            'mean_speed_mph': float(np.mean(valid)),
            'frac_congested_lt35mph': float(np.mean(valid < 35.0)),
        }
    summary = {'bin_t_s': BIN_T, 'bin_x_ft': BIN_X,
               'per_lane': per_lane,
               'wave_regime_confirmed': bool(any(
                   v['frac_congested_lt35mph'] > 0.02
                   for v in per_lane.values()))}
    json.dump(summary, open(f'{OUT}/e0_real_summary.json', 'w'), indent=2)
    log(f"E0-real done: {summary['per_lane']}")


# ------------------------------------------------------------- IDM / shared
def idm_accel(v, dv, s, T, a, b=2.0, s0=2.0):
    s_star = s0 + np.maximum(v * T + v * (-dv) / (2.0 * np.sqrt(a * b)), 0.0)
    return a * ((1.0 - (v / V0_DEFAULT) ** 4) - (s_star / np.maximum(s, .1)) ** 2)


def extract_episodes(clean, min_seconds=12.0, max_gap_ft=250.0, fps=10.0):
    d = clean[['vehicle_id', 'frame_id', 'global_time', 'lane_id', 'preceding',
               'space_hdwy', 'v_vel_s', 'v_acc_s', 'bad_acc']].copy()
    d['s_m'] = d.space_hdwy * FT2M
    ep_list, eid = [], 0
    for (vid, lid), g in d.groupby(['vehicle_id', 'lane_id'], sort=False):
        g = g.sort_values('frame_id')
        new_run = g.preceding.values != np.roll(g.preceding.values, 1)
        new_run[0] = True
        bounds = np.append(np.flatnonzero(new_run), len(g))
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            seg = g.iloc[b0:b1]
            dur = (seg.frame_id.iloc[-1] - seg.frame_id.iloc[0] + 1) / fps
            if (seg.preceding.iloc[0] == 0 or dur < min_seconds
                    or seg.s_m.max() > max_gap_ft or seg.bad_acc.any()):
                continue
            ep_list.append((eid, seg.index.to_numpy()))
            eid += 1
    frames = np.concatenate([ix for _, ix in ep_list])
    eids = np.concatenate([np.full(len(ix), e) for e, ix in ep_list])
    sub = clean.loc[frames].copy()
    sub['episode_id'] = eids
    lead_speed = clean.set_index(['vehicle_id', 'frame_id']).v_vel_s
    key = pd.MultiIndex.from_arrays([sub.preceding.values, sub.frame_id.values])
    sub['lead_vel_s'] = lead_speed.reindex(key).to_numpy()
    sub = sub.dropna(subset=['lead_vel_s'])
    sub['v_foll'] = np.clip(sub.v_vel_s.values, 0, None) * FT2M
    sub['v_lead'] = np.clip(sub.lead_vel_s.values, 0, None) * FT2M
    sub['dv'] = sub.v_lead - sub.v_foll
    sub['s'] = sub.space_hdwy * FT2M
    sub['acc'] = sub.v_acc_s * FT2M
    return sub[['episode_id', 'vehicle_id', 'global_time', 'v_foll', 'v_lead',
                'dv', 's', 'acc']]


def fit_idm(v, dv, s, acc_obs):
    def resid(p):
        T, a = p
        return idm_accel(v, dv, s, T, a) - acc_obs
    try:
        r = least_squares(resid, [1.5, 1.0], bounds=([0.2, 0.2], [5., 5.]),
                          method='trf')
    except Exception:
        return None
    if not np.all(np.isfinite(r.x)):
        return None
    return {'T': float(r.x[0]), 'a': float(r.x[1]),
            'rmse': float(np.sqrt(np.mean(r.fun ** 2)))}


def ep_features(t, v, dv, s, acc):
    """First-half summary features (detector-observable stand-in)."""
    h = len(t) // 2
    t, v, dv, s, acc = t[:h], v[:h], dv[:h], s[:h], acc[:h]
    tt = t - t[0]
    return {'mean_v': np.mean(v), 'std_v': np.std(v),
            'trend_v': np.polyfit(tt, v, 1)[0] if len(t) > 3 else 0.,
            'mean_s': np.mean(s), 'std_s': np.std(s),
            'mean_dv': np.mean(dv), 'frac_stopped': np.mean(v < 1.),
            'mean_acc': np.mean(acc)}


FEATS = ['mean_v', 'std_v', 'trend_v', 'mean_s', 'std_s',
         'mean_dv', 'frac_stopped', 'mean_acc']


# ------------------------------------------------------------------ E1
def e1_windows(eps, win_s=2.0, horizons_s=(1.0, 2.0, 4.0, 8.0), stride_s=1.0,
              fps=10.0):
    win = int(win_s * fps)
    hors = [int(h * fps) for h in horizons_s]
    stride = int(stride_s * fps)
    max_h = max(hors)
    rows = []
    for eid, e in eps.groupby('episode_id'):
        e = e.sort_values('global_time')
        n = len(e)
        if n < win + max_h + 1:
            continue
        v, vl, dv, s = (e.v_foll.values, e.v_lead.values, e.dv.values,
                        e.s.values)
        vid = int(e.vehicle_id.iloc[0])
        for t in range(win, n - max_h, stride):
            w = slice(t - win, t)
            row = {'episode_id': eid, 'vehicle_id': vid,
                  'v_now': v[t - 1], 's_now': s[t - 1], 'dv_now': dv[t - 1],
                  'v_mean': v[w].mean(), 'v_std': v[w].std(),
                  'v_slope': (v[t - 1] - v[t - win]) / win_s,
                  's_mean': s[w].mean(), 'dv_mean': dv[w].mean()}
            for h, hs in zip(hors, horizons_s):
                row[f'y_{hs}'] = v[t - 1 + h]
                row[f'vl_path_{hs}'] = vl[t - 1:t - 1 + h + 1].tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def idm_rollout(v0, s0, dv0, vl_path, T, a, b=2.0, s0_jam=2.0, dt=0.1,
               v0_freeflow=V0_DEFAULT * FT2M / FT2M):
    """Integrate IDM forward given the REALIZED leader speed path (removes
    leader-prediction error; isolates follower-response quality)."""
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


E1_FEATS = ['v_now', 's_now', 'dv_now', 'v_mean', 'v_std', 'v_slope',
           's_mean', 'dv_mean']


def e1_real(eps, idm_T, idm_a):
    horizons_s = (1.0, 2.0, 4.0, 8.0)
    W = e1_windows(eps, horizons_s=horizons_s)
    log(f'E1-real: {len(W)} windows, {W.episode_id.nunique()} episodes, '
        f'{W.vehicle_id.nunique()} vehicles')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    tr_idx, te_idx = next(gss.split(W, groups=W.vehicle_id.values))
    Wtr, Wte = W.iloc[tr_idx], W.iloc[te_idx]
    assert not (set(Wtr.vehicle_id) & set(Wte.vehicle_id))

    Xtr = Wtr[E1_FEATS].values
    results = []
    for hs in horizons_s:
        ytr, yte = Wtr[f'y_{hs}'].values, Wte[f'y_{hs}'].values
        Xte = Wte[E1_FEATS].values
        gb = GradientBoostingRegressor(random_state=0, max_depth=3,
                                       n_estimators=200).fit(Xtr, ytr)
        pred_gb = gb.predict(Xte)
        pred_persist = Wte['v_now'].values
        pred_ca = np.clip(Wte['v_now'].values + Wte['v_slope'].values * hs,
                          0.0, None)
        pred_idm = np.array([
            idm_rollout(r.v_now, r.s_now, r.dv_now, r.vl_path, idm_T, idm_a)
            for r in Wte[['v_now', 's_now', 'dv_now', f'vl_path_{hs}']]
                        .rename(columns={f'vl_path_{hs}': 'vl_path'})
                        .itertuples()
        ])

        def mape(pred):
            return float(np.mean(np.abs(pred - yte) / np.maximum(yte, 1.0)) * 100)

        results.append({'horizon_s': hs, 'mape_gb': mape(pred_gb),
                        'mape_persistence': mape(pred_persist),
                        'mape_const_accel': mape(pred_ca),
                        'mape_idm_physics': mape(pred_idm),
                        'rmse_gb': float(np.sqrt(np.mean((pred_gb - yte) ** 2))),
                        'rmse_persistence': float(np.sqrt(np.mean((pred_persist - yte) ** 2))),
                        'rmse_idm_physics': float(np.sqrt(np.mean((pred_idm - yte) ** 2)))})

    R = pd.DataFrame(results)
    R.to_csv(f'{OUT}/e1_real_results.csv', index=False)
    json.dump({'n_windows': int(len(W)), 'n_train': int(len(Wtr)),
               'n_test': int(len(Wte)), 'idm_params': {'T': idm_T, 'a': idm_a},
               'results': results}, open(f'{OUT}/e1_real_summary.json', 'w'),
              indent=2)

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
    x = np.arange(len(R)); width = 0.2
    for i, (col, lab) in enumerate([('rmse_gb', 'GB'),
                                    ('rmse_persistence', 'persist'),
                                    ('rmse_idm_physics', 'IDM physics')]):
        ax.bar(x + (i - 1) * width, R[col], width=width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f'{h:g}s' for h in R.horizon_s])
    ax.set_xlabel('horizon'); ax.set_ylabel('RMSE [m/s]')
    ax.set_title('RMSE by horizon'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_e1_real_accuracy.png', dpi=140)
    log('E1-real done:\n' + R.to_string(index=False))


# ------------------------------------------------------------------ E2
def e2_real(eps):
    rows = []
    for eid, e in eps.groupby('episode_id'):
        e = e.sort_values('global_time')
        if len(e) < MIN_FRAMES:
            continue
        h = len(e) // 2
        e2 = e.iloc[h:]                       # second half only -> target
        r = fit_idm(e2.v_foll.values, e2.dv.values, e2.s.values, e2.acc.values)
        if r:
            f = ep_features(e.global_time.values / 1000., e.v_foll.values,
                            e.dv.values, e.s.values, e.acc.values)
            rows.append({'episode_id': eid, 'vehicle_id': int(e.vehicle_id.iloc[0]),
                        **f, 'T': r['T'], 'a': r['a'], 'fit_rmse': r['rmse']})
    F = pd.DataFrame(rows)
    F = F[F.fit_rmse < 5.0].reset_index(drop=True)
    log(f'E2-real: {len(F)} usable episodes (causal target fit, '
        f'{F.vehicle_id.nunique()} vehicles)')

    X, Y, groups = F[FEATS].values, F[['T', 'a']].values, F.vehicle_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=0)
    tr_idx, te_idx = next(gss.split(X, Y, groups=groups))
    Xtr, Xte, Ytr, Yte = X[tr_idx], X[te_idx], Y[tr_idx], Y[te_idx]
    assert not (set(groups[tr_idx]) & set(groups[te_idx])), 'vehicle leakage'
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    groups_tr = groups[tr_idx]
    rng = np.random.RandomState(0)
    B = 40
    results, preds = {}, {}
    for j, tgt in enumerate(['T', 'a']):
        ytr, yte = Ytr[:, j], Yte[:, j]
        ridge_pt = Ridge(alpha=10.).fit(Ztr, ytr)
        gb_pt = GradientBoostingRegressor(random_state=0).fit(Ztr, ytr)
        rmse_ridge = float(np.sqrt(np.mean((ridge_pt.predict(Zte) - yte) ** 2)))
        rmse_gb = float(np.sqrt(np.mean((gb_pt.predict(Zte) - yte) ** 2)))

        gss_inner = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=1)
        fit_ix, cal_ix = next(gss_inner.split(Ztr, ytr, groups=groups_tr))
        p_cal = np.empty((B, len(cal_ix))); p_te = np.empty((B, len(yte)))
        for b in range(B):
            ix = rng.choice(fit_ix, len(fit_ix), replace=True)
            m = Ridge(alpha=10.).fit(Ztr[ix], ytr[ix])
            p_cal[b] = m.predict(Ztr[cal_ix]); p_te[b] = m.predict(Zte)
        ens_mean, ens_std = p_te.mean(0), p_te.std(0)
        cal_std = np.maximum(p_cal.std(0), 1e-3)
        rmse_ens = float(np.sqrt(np.mean((ens_mean - yte) ** 2)))

        cov_boot = {}
        for nom, z in [(0.80, 1.2816), (0.95, 1.9600)]:
            lo, hi = ens_mean - z * ens_std, ens_mean + z * ens_std
            cov_boot[nom] = float(np.mean((yte >= lo) & (yte <= hi)))

        scores = np.abs(ytr[cal_ix] - p_cal.mean(0)) / cal_std
        cov_conf, width_conf, ci_lo, ci_hi = {}, {}, None, None
        te_std = np.maximum(ens_std, 1e-3)
        for nom in (0.80, 0.95):
            k = int(np.ceil((len(scores) + 1) * nom))
            q = np.sort(scores)[min(k, len(scores)) - 1]
            lo, hi = ens_mean - q * te_std, ens_mean + q * te_std
            cov_conf[nom] = float(np.mean((yte >= lo) & (yte <= hi)))
            width_conf[nom] = float(np.mean(hi - lo))
            if nom == 0.95:
                ci_lo, ci_hi = lo, hi
        results[tgt] = {'rmse_point_ridge': rmse_ridge, 'rmse_point_gb': rmse_gb,
                        'rmse_ensemble': rmse_ens,
                        'coverage_bootstrap_raw': cov_boot,
                        'coverage_conformal': cov_conf,
                        'width_conformal_95': width_conf[0.95],
                        'target_std': float(yte.std())}
        preds[tgt] = (yte, ens_mean, ci_lo, ci_hi)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    mks = ['o', 's']
    for k, tgt in enumerate(['T', 'a']):
        yt, pm, plo, phi = preds[tgt]
        ax.errorbar(yt, pm, yerr=[pm - plo, phi - pm], fmt=mks[k],
                    ms=3, lw=.6, alpha=.45, label=f'{tgt} (95% conf. CI)')
    ax.plot([0, 5.2], [0, 5.2], 'k--', lw=1); ax.set_xlim(0, 5.2); ax.set_ylim(0, 5.2)
    ax.set_xlabel('true $\\hat\\theta$ (second-half-only IDM fit)')
    ax.set_ylabel('prediction'); ax.legend(fontsize=8)
    ax.set_title('E2-real: calibrated parameter recovery (real NGSIM)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_e2_real_scatter.png', dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    noms = [0.80, 0.95]
    for ax, key, ttl in [(axes[0], 'coverage_bootstrap_raw',
                          'raw bootstrap (model variance only)'),
                         (axes[1], 'coverage_conformal',
                          'scaled split-conformal')]:
        for k, tgt in enumerate(['T', 'a']):
            ax.plot(noms, [results[tgt][key][n] for n in noms],
                    mks[k] + '-', label=tgt)
        ax.plot(noms, noms, 'k--', lw=1, label='nominal')
        ax.set_xlabel('nominal coverage'); ax.set_title(ttl, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('empirical coverage'); axes[0].legend(fontsize=9)
    fig.suptitle('E2-real: interval calibration on real NGSIM data (vehicle-grouped split)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_e2_real_coverage.png', dpi=140)

    csv = pd.DataFrame([{'target': t, **results[t]} for t in ['T', 'a']])
    csv.to_csv(f'{OUT}/e2_real_results.csv', index=False)
    json.dump({'n_train_vehicles': int(len(set(groups[tr_idx]))),
               'n_test_vehicles': int(len(set(groups[te_idx]))),
               'split': 'grouped_by_vehicle_id',
               'target_fit_window': 'second_half_of_episode_only',
               'results': results},
              open(f'{OUT}/e2_real_summary.json', 'w'), indent=2)
    log(f"E2-real done: T cov80 conf={results['T']['coverage_conformal'][0.8]:.3f}, "
        f"a cov80 conf={results['a']['coverage_conformal'][0.8]:.3f}")
    return F  # for E1's population-median IDM params


# ------------------------------------------------------------------ E4
def _movavg(a, k=5):
    filled = np.where(np.isnan(a), 0., a)
    valid = (~np.isnan(a)).astype(float)
    num = uniform_filter1d(filled, size=k, axis=0, mode='nearest') * k
    den = uniform_filter1d(valid, size=k, axis=0, mode='nearest') * k
    return num / np.maximum(den, 1.)


def build_cells(clean):
    d = clean[clean.lane_id.isin(MAINLINE_LANES)].copy()
    d['t_bin'] = ((d.global_time - d.global_time.min()) / 1000. // BIN_T).astype(int)
    d['x_bin'] = (d.local_y_s // BIN_X).astype(int)
    d['cell'] = d.x_bin // CELL_XBINS
    agg = d.groupby(['lane_id', 't_bin', 'cell']).v_vel_s.agg(['mean', 'size'])
    agg = agg[agg['size'] >= 2] * 0.681818
    out = {}
    for lane, g in agg.groupby(level=0):
        piv = g.droplevel(0)['mean'].unstack('cell')
        piv = piv.reindex(index=pd.RangeIndex(int(piv.index.max()) + 1),
                          columns=pd.RangeIndex(int(piv.columns.max()) + 1))
        out[int(lane)] = piv.sort_index(axis=1).interpolate(axis=0, limit=5)
    return out


def label_featurize(cells):
    hb, tb, rb = int(HORIZON_S / BIN_T), int(TRAIL_S / BIN_T), int(20. / BIN_T)
    rows = []
    for lane in sorted(cells):
        V = cells[lane].to_numpy()
        n_t, n_c = V.shape
        Vs = _movavg(V, k=5)
        for c in range(n_c):
            v, vs = V[:, c], Vs[:, c]
            if np.all(np.isnan(v)):
                continue
            cu = max(c - 1, 0)
            v_up, v_ups, v_lane = V[:, cu], Vs[:, cu], np.nanmean(V, axis=1)
            onset = np.zeros(n_t, bool)
            for t in range(rb, n_t - hb):
                ref = np.nanmean(v[t - rb:t])
                fut = vs[t + 1:t + 1 + hb]
                if not np.isnan(fut).any() and np.nanmin(fut) <= ref - DROP_MPH:
                    onset[t] = True
            for t in range(tb, n_t - hb, int(SAMPLE_EVERY / BIN_T)):
                w, ws_, wu, wl = (v[t-tb:t], vs[t-tb:t], v_up[t-tb:t],
                                  v_lane[t-tb:t])
                if np.isnan(w).any() or np.isnan(wl).any():
                    continue
                tt = np.arange(tb) * BIN_T
                rows.append({'lane': lane, 'cell': c, 't_sec': t * BIN_T,
                             'f_mean': np.nanmean(w), 'f_std': np.nanstd(ws_),
                             'f_slope': np.polyfit(tt, w, 1)[0],
                             'f_min': np.nanmin(w),
                             'f_up_mean': np.nanmean(wu),
                             'f_up_min': np.nanmin(wu),
                             'f_lane_mean': np.nanmean(wl),
                             'f_lane_slope': np.polyfit(tt, wl, 1)[0],
                             'y': int(onset[t])})
    return pd.DataFrame(rows)


EF = ['f_mean', 'f_std', 'f_slope', 'f_min',
      'f_up_mean', 'f_up_min', 'f_lane_mean', 'f_lane_slope']


def evaluate(tr, te, tag):
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(tr[EF].values, tr.y.values)
    p = clf.predict_proba(te[EF].values)[:, 1]
    pers = (te.f_mean.values - te.f_up_min.values >= 8.).astype(float)
    return {'split': tag, 'n_train': int(len(tr)), 'n_test': int(len(te)),
            'base_rate': float(te.y.mean()),
            'auroc_gb': float(roc_auc_score(te.y, p)),
            'auprc_gb': float(average_precision_score(te.y, p)),
            'auroc_persistence': float(roc_auc_score(te.y, pers)),
            'auprc_majority': float(te.y.mean())}


def e4_real(clean):
    S = label_featurize(build_cells(clean))
    # temporal split: drop training rows whose forward label horizon
    # extends past the split boundary (would peek at test-period dynamics)
    tr_t = S[(S.t_sec < TEMPORAL_SPLIT_S) &
             (S.t_sec + HORIZON_S <= TEMPORAL_SPLIT_S)]
    te_t = S[S.t_sec >= TEMPORAL_SPLIT_S]
    res = [evaluate(tr_t, te_t, 'temporal_holdout'),
           evaluate(S[S.lane.isin([1, 2, 3])],
                    S[S.lane.isin([4, 5])], 'lane_holdout')]
    R = pd.DataFrame(res)
    R.to_csv(f'{OUT}/e4_real_results.csv', index=False)
    json.dump({'n_samples': int(len(S)), 'onset_rate': float(S.y.mean()),
               'results': res},
              open(f'{OUT}/e4_real_summary.json', 'w'), indent=2)

    tags = ['temporal_holdout', 'lane_holdout']
    x = np.arange(len(tags))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i, (m, lab, col) in enumerate([('auroc_gb', 'gradient boosting', '#2c7fb8'),
                                       ('auroc_persistence', 'persistence', '#bbbbbb')]):
        bars = ax.bar(x + (i - .5) * .3, [r[m] for r in res], width=.28, color=col)
        for rect, val in zip(bars, [r[m] for r in res]):
            ax.text(rect.get_x() + rect.get_width() / 2, val + .01, f'{val:.2f}',
                    ha='center', fontsize=9)
    ax.axhline(.5, color='k', ls='--', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel('AUROC'); ax.set_ylim(0, 1)
    ax.set_title('E4-real: wave-onset prediction (real NGSIM)')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c)
                       for c in ['#2c7fb8', '#bbbbbb']],
              labels=['gradient boosting', 'persistence'], fontsize=8)
    ax = axes[1]
    ax.bar(x - .15, [r['auprc_gb'] for r in res], width=.28, color='#2c7fb8',
           label='GB AUPRC')
    ax.bar(x + .15, [r['base_rate'] for r in res], width=.28, color='#bbbbbb',
           label='base rate')
    ax.set_xticks(x); ax.set_xticklabels(tags); ax.set_ylabel('AUPRC')
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title('precision-recall vs class imbalance')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_e4_real_roc.png', dpi=140)
    log('E4-real done:\n' + R.to_string(index=False))


def main():
    log('loading + cleaning NGSIM US-101 ...')
    df = load_ngsim(NGSIM_TXT)
    clean = clean_trajectories(df)
    log(f'{len(clean):,} rows, {clean.vehicle_id.nunique()} vehicles, '
        f'{int(clean.bad_acc.sum()):,} swap-flagged frames dropped')
    e0_real(clean)
    eps = extract_episodes(clean)
    F = e2_real(eps)
    idm_T, idm_a = float(F['T'].median()), float(F['a'].median())
    e1_real(eps, idm_T, idm_a)
    e4_real(clean)
    log('ALL REAL-DATA EXPERIMENTS COMPLETE')


if __name__ == '__main__':
    main()
