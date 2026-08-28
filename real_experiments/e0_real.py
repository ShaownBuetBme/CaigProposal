"""E0-real -- Sanity gate on REAL data: does the cleaned NGSIM US-101 field
actually exhibit stop-and-go waves before we let E4 label them?

Builds a lane-level space-time speed field (2 s x 100 ft bins, mainline
lanes 1-5), renders it as heatmaps, and quantifies wave activity
(share of congested bins, temporal speed variability, detected
upstream-moving wave fronts).

Output: fig_e0_real_speedfield.png, e0_real_summary.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ngsim_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'experiment_results')
CACHE = os.path.join(HERE, 'cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

MAINLINE_LANES = [1, 2, 3, 4, 5]
BIN_T = 2.0        # s
BIN_X = 100.0      # ft
CONGESTED_MPH = 35.0


def build_field(clean):
    """Return dict lane -> 2D array [time_bin, x_bin] of mean speed (mph)."""
    d = clean[clean.lane_id.isin(MAINLINE_LANES)].copy()
    d['t_bin'] = ((d.global_time - d.global_time.min()) / 1000.0 // BIN_T).astype(int)
    d['x_bin'] = (d.local_y_s // BIN_X).astype(int)
    fields = {}
    for lane, g in d.groupby('lane_id'):
        agg = g.groupby(['t_bin', 'x_bin']).v_vel_s.agg(['mean', 'size'])
        agg = agg[agg['size'] >= 2]                    # denoise sparse bins
        piv = agg['mean'].unstack('x_bin') * 0.681818  # ft/s -> mph
        fields[lane] = piv
    return fields


def count_wave_fronts(field):
    """Rough count of upstream-moving fronts: sign changes of the temporal
    derivative of the lane-mean speed from falling to rising while below
    free-flow, tracked on the x-averaged profile."""
    prof = field.mean(axis=0)                     # average over space? no --
    # field is [t, x]; take spatial mean per t
    prof = np.nanmean(field, axis=1)
    prof = pd.Series(prof).interpolate().values
    dv = np.diff(prof)
    fronts = int(np.sum((dv[:-1] < -2.0) & (dv[1:] > -1.0)))
    return fronts


def main():
    cache_f = os.path.join(CACHE, 'clean.parquet')
    if os.path.exists(cache_f):
        clean = pd.read_parquet(cache_f)
    else:
        df = nc.load_ngsim(os.path.join(HERE, '..', 'real_data', 'ngsim',
                                        'trajectories-0750am-0805am.txt'))
        clean, _ = nc.clean_trajectories(df)
        clean.to_parquet(cache_f)

    fields = build_field(clean)
    t_bins = max(f.index.max() for f in fields.values()) + 1
    x_bins = max(f.columns.max() for f in fields.values()) + 1

    grid = np.full((len(fields), t_bins, x_bins), np.nan)
    for k, (lane, f) in enumerate(sorted(fields.items())):
        grid[k][np.ix_(f.index.to_numpy(), f.columns.to_numpy())] = f.to_numpy()

    # ------------------------------- figure ---------------------------------
    fig, axes = plt.subplots(len(grid), 1, figsize=(11, 2.1 * len(grid)),
                             sharex=True, sharey=True)
    extent = [0, x_bins * BIN_X, t_bins * BIN_T, 0]
    for k, ax in enumerate(axes):
        im = ax.imshow(grid[k], aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=70, extent=extent)
        ax.set_ylabel(f'lane {sorted(fields)[k]}\ntime [s]')
        plt.colorbar(im, ax=ax, pad=0.01, label='speed [mph]')
    axes[-1].set_xlabel('space along US-101 [ft]')
    fig.suptitle('E0-real: NGSIM US-101 space-time speed field (07:50-08:05)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_e0_real_speedfield.png'), dpi=140)

    # ------------------------------ metrics ---------------------------------
    summary = {'bins_t': BIN_T, 'bins_x_ft': BIN_X,
               'congestion_threshold_mph': CONGESTED_MPH}
    per_lane = {}
    for lane in sorted(fields):
        arr = grid[list(sorted(fields)).index(lane)]
        valid = arr[~np.isnan(arr)]
        frac_cong = float(np.mean(valid < CONGESTED_MPH))
        tprof = pd.Series(np.nanmean(arr, axis=1)).interpolate().values
        summary_lane = {
            'mean_speed_mph': float(np.mean(valid)),
            'std_in_time_of_lane_mean_mph': float(np.std(tprof)),
            'frac_congested': frac_cong,
            'wave_fronts_detected': count_wave_fronts(arr),
        }
        per_lane[str(lane)] = summary_lane
    summary['per_lane'] = per_lane
    summary['any_lane_wave_activity'] = bool(
        any(v['frac_congested'] > 0.02 for v in per_lane.values()))
    with open(os.path.join(OUT, 'e0_real_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
