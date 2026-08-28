"""E2-real -- Uncertainty-aware IDM calibration on REAL NGSIM US-101 data.

Design (mirrors pilot E2 at real-data scale), CAUSAL and GROUPED to avoid two
leakage failure modes found in a methodology audit (2026-08-23):
  * Ground-truth targets: per-episode IDM parameters theta=(T, a) fitted by
    trajectory-level least squares on the SECOND HALF of each following
    episode ONLY (b, s0 fixed at literature values -- see
    ngsim_common.fit_idm_episode). Fitting on the full episode would let the
    target "see" the same first-half data the features are computed from;
    a full-vs-second-half-only refit check found only corr=0.73/0.51 (T/a)
    agreement, so this is not a cosmetic change.
  * Observations available to the calibrator: aggregate summary statistics
    computed from the FIRST HALF of the episode only (the stand-in for what a
    traffic-management center's detectors would see) -- now strictly disjoint
    in time from the target-fitting window.
  * Train/test split is GROUPED BY VEHICLE ID, not by episode: 37% of
    episodes (858/2315) share a vehicle with >=1 other episode (a driver
    passing through multiple following relationships), so a random
    episode-level split leaks driver-specific style across the split.
  * Estimators compared on held-out episodes:
      - point baseline: ridge regression  S -> theta_hat
      - uncertainty-aware: bootstrap-bagged ridge ensemble -> mean +/- std,
        Gaussian predictive intervals at 80 % and 95 %
  * Metrics: held-out RMSE per parameter, interval coverage and mean width.

Outputs: e2_real_results.csv, fig_e2_real_coverage.png,
         fig_e2_real_scatter.png, e2_real_summary.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

import ngsim_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'experiment_results')
CACHE = os.path.join(HERE, 'cache')
NGSIM_TXT = os.path.join(HERE, '..', 'real_data', 'ngsim',
                         'trajectories-0750am-0805am.txt')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

B_BOOT = 40          # bootstrap ensemble size
RMSE_FIT_MAX = 5.0   # drop episodes whose own IDM fit is degenerate
MIN_FRAMES = 40      # need enough frames in EACH half for a stable fit


def build_dataset():
    """Returns (eps, fits, F) where fits/F use only the causal (second-half)
    IDM fit as ground truth, and F also carries vehicle_id for grouping."""
    cache_eps = os.path.join(CACHE, 'episodes.parquet')
    cache_fits = os.path.join(CACHE, 'fits_causal.parquet')
    if os.path.exists(cache_eps):
        eps = pd.read_parquet(cache_eps)
    else:
        df = nc.load_ngsim(NGSIM_TXT)
        clean, _ = nc.clean_trajectories(df)
        eps = nc.extract_following_episodes(clean)
        eps.to_parquet(cache_eps)

    if os.path.exists(cache_fits):
        fits = pd.read_parquet(cache_fits)
    else:
        rows = []
        for eid, e in eps.groupby('episode_id'):
            e = e.sort_values('global_time')
            if len(e) < MIN_FRAMES:
                continue
            h = len(e) // 2
            e2 = e.iloc[h:]                      # second half only -> target
            r = nc.fit_idm_episode(e2.global_time.values / 1000.,
                                   e2.v_foll.values, e2.dv.values,
                                   e2.s.values, e2.acc.values)
            if r:
                rows.append({'episode_id': eid,
                            'vehicle_id': int(e.vehicle_id.iloc[0]), **r})
        fits = pd.DataFrame(rows)
        fits.to_parquet(cache_fits)
    return eps, fits


def feature_table(eps):
    rows = []
    for eid, e in eps.groupby('episode_id'):
        e = e.sort_values('global_time')
        if len(e) < MIN_FRAMES:
            continue
        h = len(e) // 2
        f = nc.episode_summary_features(
            e.global_time.values[:h] / 1000., e.v_foll.values[:h],
            e.dv.values[:h], e.s.values[:h], e.acc.values[:h])
        rows.append({'episode_id': eid, **f})
    return pd.DataFrame(rows).set_index('episode_id')


def main():
    eps, fits = build_dataset()
    good = fits[fits.rmse < RMSE_FIT_MAX]
    F = feature_table(eps).join(
        good.set_index('episode_id')[['vehicle_id', 'T', 'a']], how='inner')
    feats = list(nc.EPISODE_FEATURE_NAMES)
    X, Y = F[feats].values, F[['T', 'a']].values
    groups = F['vehicle_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    tr_idx, te_idx = next(gss.split(X, Y, groups=groups))
    Xtr, Xte, Ytr, Yte = X[tr_idx], X[te_idx], Y[tr_idx], Y[te_idx]
    train_vehicles = set(groups[tr_idx])
    test_vehicles = set(groups[te_idx])
    assert not (train_vehicles & test_vehicles), 'vehicle leakage across split'
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    groups_tr = groups[tr_idx]
    rng = np.random.RandomState(0)
    n_tr = len(Ztr)

    results = {}
    preds_ens = {}
    for j, tgt in enumerate(['T', 'a']):
        ytr, yte = Ytr[:, j], Yte[:, j]

        # --- point baselines -------------------------------------------------
        ridge_pt = Ridge(alpha=10.).fit(Ztr, ytr)
        gb_pt = GradientBoostingRegressor(random_state=0).fit(Ztr, ytr)
        rmse_ridge = float(np.sqrt(np.mean((ridge_pt.predict(Zte) - yte) ** 2)))
        rmse_gb = float(np.sqrt(np.mean((gb_pt.predict(Zte) - yte) ** 2)))

        # --- bootstrap-bagged ridge ensemble ---------------------------------
        # fit/calibration split is ALSO grouped by vehicle_id (same reason as
        # the outer train/test split)
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.30,
                                      random_state=1)
        fit_idx, cal_idx = next(gss_inner.split(Ztr, ytr, groups=groups_tr))
        boot_pred_cal = np.empty((B_BOOT, len(cal_idx)))
        boot_pred_te = np.empty((B_BOOT, len(yte)))
        for b in range(B_BOOT):
            idx = rng.choice(fit_idx, len(fit_idx), replace=True)
            m = Ridge(alpha=10.).fit(Ztr[idx], ytr[idx])
            boot_pred_cal[b] = m.predict(Ztr[cal_idx])
            boot_pred_te[b] = m.predict(Zte)
        ens_mean = boot_pred_te.mean(0)
        ens_std = boot_pred_te.std(0)
        rmse_ens = float(np.sqrt(np.mean((ens_mean - yte) ** 2)))
        cal_mean = boot_pred_cal.mean(0)
        cal_std = np.maximum(boot_pred_cal.std(0), 1e-3)

        # raw Gaussian bootstrap intervals (model-variance-only -- expected
        # to be badly miscalibrated on real data; kept for the comparison)
        cov_boot = {}
        for nom in (0.80, 0.95):
            z = {0.80: 1.2816, 0.95: 1.9600}[nom]
            lo, hi = ens_mean - z * ens_std, ens_mean + z * ens_std
            cov_boot[nom] = float(np.mean((yte >= lo) & (yte <= hi)))

        # scaled split-conformal intervals (Vovk/Papadopoulos): absolute
        # residuals on the calibration fold, normalised by the ensemble's own
        # dispersion -> finite-sample marginal coverage >= nominal.
        scores = np.abs(ytr[cal_idx] - cal_mean) / cal_std
        cov_conf, width_conf = {}, {}
        conf_lo = np.empty_like(yte); conf_hi = np.empty_like(yte)
        for nom in (0.80, 0.95):
            k = int(np.ceil((len(scores) + 1) * nom))
            q = np.sort(scores)[min(k, len(scores)) - 1]
            lo = cal_mean - q * cal_std          # predict cal-style on test
            hi = cal_mean + q * cal_std
            # recompute with test-set ensemble stats
            te_std = np.maximum(ens_std, 1e-3)
            lo = ens_mean - q * te_std
            hi = ens_mean + q * te_std
            conf_lo, conf_hi = lo, hi
            cov_conf[nom] = float(np.mean((yte >= lo) & (yte <= hi)))
            width_conf[nom] = float(np.mean(hi - lo))

        results[tgt] = {
            'rmse_point_ridge': rmse_ridge,
            'rmse_point_gb': rmse_gb,
            'rmse_ensemble': rmse_ens,
            'coverage_bootstrap_raw': cov_boot,
            'coverage_conformal': cov_conf,
            'mean_interval_width_conformal': width_conf,
            'target_std': float(yte.std()),
        }
        preds_ens[tgt] = (yte, ens_mean, conf_hi - ens_mean, conf_lo)

    # ----------------------------- figures ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    labels, xs = [], []
    for k, tgt in enumerate(['T', 'a']):
        r = results[tgt]
        vals = [r['rmse_point_gb'], r['rmse_point_ridge'], r['rmse_ensemble']]
        xs.append([k - 0.22, k, k + 0.22])
        labels.extend(['GB point', 'Ridge point', 'Bagged ensemble'])
        bars = ax.bar([k - 0.22, k, k + 0.22], vals, width=0.2,
                      color=['#bbbbbb', '#888888', '#2c7fb8'])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['desired headway T [s]', 'max accel a [m/s$^2$]'])
    ax.set_ylabel('held-out RMSE (real NGSIM episodes)')
    ax.set_title('E2-real: calibration accuracy')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c)
                       for c in ['#bbbbbb', '#888888', '#2c7fb8']],
              labels=['GB point', 'Ridge point', 'Bagged ensemble'],
              fontsize=8)

    ax = axes[1]
    mks = ['o', 's']
    for k, tgt in enumerate(['T', 'a']):
        yt, pm, ps, plo = preds_ens[tgt]
        ax.errorbar(yt, pm, yerr=[pm - plo, ps], fmt=mks[k], ms=3, lw=0.6,
                    alpha=0.45, label=f'{tgt} (95% conformal CI)')
    lims = [0, 5.2]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('true $\\hat\\theta$ (second-half-only IDM fit)')
    ax.set_ylabel('ensemble prediction')
    ax.set_title('E2-real: predictions with uncertainty')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_e2_real_scatter.png'), dpi=140)

    # coverage plot: raw bootstrap vs scaled conformal
    noms = [0.80, 0.95]
    fig, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, ttl in [(axes2[0], 'coverage_bootstrap_raw',
                          'raw bootstrap (model variance only)'),
                         (axes2[1], 'coverage_conformal',
                          'scaled split-conformal')]:
        for k, tgt in enumerate(['T', 'a']):
            covs = [results[tgt][key][n] for n in noms]
            ax.plot(noms, covs, mks[k] + '-', label=tgt)
        ax.plot(noms, noms, 'k--', lw=1, label='nominal')
        ax.set_xlabel('nominal coverage'); ax.set_title(ttl, fontsize=10)
        ax.set_ylim(0, 1)
    axes2[0].set_ylabel('empirical coverage')
    axes2[0].legend(fontsize=9)
    fig.suptitle('E2-real: interval calibration on real NGSIM data')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_e2_real_coverage.png'), dpi=140)

    # ----------------------------- artifacts -------------------------------
    rows_csv = []
    for tgt in ['T', 'a']:
        r = results[tgt]
        rows_csv.append({'target': tgt,
                         'rmse_point_gb': r['rmse_point_gb'],
                         'rmse_point_ridge': r['rmse_point_ridge'],
                         'rmse_ensemble': r['rmse_ensemble'],
                         'cov_bootstrap_80': r['coverage_bootstrap_raw'][0.80],
                         'cov_bootstrap_95': r['coverage_bootstrap_raw'][0.95],
                         'cov_conformal_80': r['coverage_conformal'][0.80],
                         'cov_conformal_95': r['coverage_conformal'][0.95],
                         'width_conformal_80':
                             r['mean_interval_width_conformal'][0.80],
                         'width_conformal_95':
                             r['mean_interval_width_conformal'][0.95],
                         'target_std': r['target_std']})
    pd.DataFrame(rows_csv).to_csv(os.path.join(OUT, 'e2_real_results.csv'),
                                  index=False)
    summary = {'n_episodes_total': int(len(fits)),
               'n_episodes_used': int(len(F)),
               'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
               'n_train_vehicles': len(train_vehicles),
               'n_test_vehicles': len(test_vehicles),
               'split': 'grouped_by_vehicle_id',
               'target_fit_window': 'second_half_of_episode_only',
               'bootstrap_size': B_BOOT,
               'fixed_params': {'b': 2.0, 's0': 2.0},
               'results': results}
    with open(os.path.join(OUT, 'e2_real_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
