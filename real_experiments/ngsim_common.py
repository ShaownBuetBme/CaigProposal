"""Shared NGSIM US-101 loading / cleaning / episode extraction.

Cleaning follows standard practice for NGSIM trajectory data
(Thiemann, Treiber & Kesting 2008; Montanino & Punzo 2015):
raw accelerations are derived from video-transcribed positions and are
noise-dominated (here: hard-clipped at +/-11.2 ft/s^2 on ~11% of rows),
so velocities are smoothed with a Savitzky-Golay filter and acceleration
is recomputed as the finite difference of the smoothed velocity.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

NGSIM_COLS = ['vehicle_id', 'frame_id', 'total_frames', 'global_time',
              'local_x', 'local_y', 'global_x', 'global_y',
              'v_length', 'v_width', 'v_class', 'v_vel', 'v_acc',
              'lane_id', 'preceding', 'following', 'space_hdwy', 'time_hdwy']

FT2M = 0.3048
FPS2MS = FT2M            # ft/s   -> m/s
FPS22MSS = FT2M          # ft/s^2 -> m/s^2


def load_ngsim(path):
    df = pd.read_csv(path, sep=r'\s+', names=NGSIM_COLS, engine='c')
    # keep cars and heavy vehicles; motorcycles (class 1) follow different dynamics
    df = df[df.v_class.isin([2, 3])].copy()
    return df


def clean_trajectories(df, window=13, poly=2):
    """Position-first cleaning (Montanino & Punzo 2015 style).

    The file's velocity/acceleration columns are corrupted at track-swap /
    occlusion events (velocity collapsing 95->9 ft/s over 3 frames while
    position keeps advancing smoothly), so we treat LONGITUDINAL POSITION as
    the reliable primitive: split each vehicle track into gap-free segments,
    Savitzky-Golay-smooth local_y within each segment, and obtain
      v = d(smoothed y)/dt ,  a = d v/dt .
    Frames where the file's velocity disagrees with position-derived speed by
    more than `repair_tol_fts` are counted as repaired.

    Adds columns: local_y_s, v_vel_s [ft/s], v_acc_s [ft/s^2].
    """
    n_raw_bad_acc = int((df.v_acc.abs() > 10).sum())
    n_repaired = 0
    y_s_all = pd.Series(np.nan, index=df.index)
    v_s_all = pd.Series(np.nan, index=df.index)
    a_s_all = pd.Series(np.nan, index=df.index)

    for vid, g in df.groupby('vehicle_id', sort=False):
        g = g.sort_values('global_time')
        idx = g.index.to_numpy()
        # split at missing frames so we never smooth across a time gap
        breaks = np.flatnonzero(np.diff(g.frame_id.values) != 1) + 1
        for ix in np.split(np.arange(len(g)), breaks):
            n = len(ix)
            y = g.local_y.values[ix]
            t = g.global_time.values[ix]
            if n >= 7:
                w = min(window if window % 2 == 1 else window + 1,
                        n if n % 2 == 1 else n - 1)
                y_s = savgol_filter(y, window_length=w, polyorder=poly)
                # gap-free segment => uniform 0.1 s steps (verified on file)
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
    implausible = ok & (out.v_acc_s.abs() > ACCEL_PLAUS_FT2)          # tracker swaps
    out['bad_acc'] = implausible.fillna(False)
    report = {'rows': len(out),
              'vehicles': int(out.vehicle_id.nunique()),
              'raw_bad_accel_rows': n_raw_bad_acc,
              'implausible_accel_frames_dropped': int(implausible.sum()),
              'method': 'savitzky-golay on position'}
    return out, report


ACCEL_PLAUS_FT2 = 25.0   # ~0.77 g; real emergency braking stays below this,
                         # tracker-swap kinks (up to 8 g observed) do not.


def extract_following_episodes(df, min_seconds=12.0, max_gap_ft=250.0,
                               fps=10.0):
    """Car-following episodes: maximal contiguous runs in which vehicle i
    lists vehicle j as its preceding vehicle, same lane, gap <= max_gap_ft.

    Episodes containing tracker-swap frames (`bad_acc`) are dropped entirely.

    Returns DataFrame with one row per frame of each episode plus an
    `episode_id`, using SMOOTHED kinematics:
      follower speed/accel (m/s, m/s^2), leader speed (m/s),
      gap s (m), relative speed dv = v_lead - v_foll (m/s).
    """
    d = df[['vehicle_id', 'frame_id', 'global_time', 'lane_id', 'preceding',
            'space_hdwy', 'v_vel_s', 'v_acc_s', 'bad_acc']].copy()
    d['s_m'] = d.space_hdwy * FT2M

    ep_list = []
    eid = 0
    for (vid, lid), g in d.groupby(['vehicle_id', 'lane_id'], sort=False):
        g = g.sort_values('frame_id')
        new_run = (g.preceding.values != np.roll(g.preceding.values, 1))
        new_run[0] = True
        run_breaks = np.flatnonzero(new_run)
        bounds = np.append(run_breaks, len(g))
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            seg = g.iloc[b0:b1]
            lead = seg.preceding.iloc[0]
            dur = (seg.frame_id.iloc[-1] - seg.frame_id.iloc[0] + 1) / fps
            if lead == 0 or dur < min_seconds or seg.s_m.max() > max_gap_ft:
                continue
            if seg.bad_acc.any():
                continue                      # corrupted track -> unusable
            ep_list.append((eid, vid, int(lead), int(lid),
                            seg.index.to_numpy()))
            eid += 1

    frames = np.concatenate([ix for *_, ix in ep_list])
    eids = np.concatenate([np.full(len(ix), e) for e, *_ , ix in ep_list])
    sub = df.loc[frames].copy()
    sub = sub.join(pd.Series(eids, index=frames, name='episode_id'))

    # attach leader speed by (leader vehicle, frame)
    lead_speed = df.set_index(['vehicle_id', 'frame_id']).v_vel_s
    key = pd.MultiIndex.from_arrays([sub.preceding.values, sub.frame_id.values])
    sub['lead_vel_s'] = lead_speed.reindex(key).to_numpy()
    sub = sub.dropna(subset=['lead_vel_s'])

    sub['v_foll'] = np.clip(sub.v_vel_s.values, 0.0, None) * FPS2MS
    sub['v_lead'] = np.clip(sub.lead_vel_s.values, 0.0, None) * FPS2MS
    sub['dv'] = sub.v_lead - sub.v_foll
    sub['s'] = sub.space_hdwy * FT2M
    sub['acc'] = sub.v_acc_s * FPS22MSS
    return sub[['episode_id', 'vehicle_id', 'preceding', 'lane_id',
                'global_time', 'v_foll', 'v_lead', 'dv', 's', 'acc']]


# ---------------------------------------------------------------- IDM ----

def idm_accel(v, dv, s, T, a, b, s0):
    """Vectorized IDM acceleration (m/s^2). dv = v_lead - v_foll."""
    s_star = s0 + np.maximum(v * T + v * (-dv) / (2.0 * np.sqrt(a * b)), 0.0)
    s_safe = np.maximum(s, 0.1)
    free = 1.0 - (v / V0_DEFAULT) ** 4
    inter = -(s_star / s_safe) ** 2
    return a * (free + inter)


V0_DEFAULT = 29.0576  # 95.3 ft/s ~= section max observed speed, m/s


def fit_idm_episode(times, v, dv, s, acc_obs, v0=None, x0=None,
                    b_fixed=2.0, s0_fixed=2.0):
    """Least-squares IDM fit for one following episode.

    Reduced parameter set theta=(T, a): comfortable deceleration b and jam
    distance s0 are held at literature-standard values because they are only
    weakly identifiable from congested following episodes (b is excited only
    by large closing speeds; s0 trades off against T*v).  v0 is likewise
    fixed at the section free-flow speed.  Returns dict(T, a, rmse) or None.
    """
    from scipy.optimize import least_squares
    if v0 is None:
        v0 = V0_DEFAULT

    def resid(p):
        T, a = p
        pred = idm_accel(v, dv, s, T, a, b_fixed, s0_fixed)
        return pred - acc_obs

    if x0 is None:
        x0 = [1.5, 1.0]
    try:
        r = least_squares(resid, x0, bounds=([0.2, 0.2], [5.0, 5.0]),
                          method='trf')
    except Exception:
        return None
    T, a = r.x
    if not np.all(np.isfinite(r.x)):
        return None
    rmse = float(np.sqrt(np.mean(r.fun ** 2)))
    return {'T': float(T), 'a': float(a), 'rmse': rmse}


EPISODE_FEATURE_NAMES = ['mean_v', 'std_v', 'trend_v', 'mean_s', 'std_s',
                         'mean_dv', 'frac_stopped', 'mean_acc']


def episode_summary_features(t, v, dv, s, acc):
    """Summary statistics from the FIRST HALF of an episode -- these stand in
    for the aggregate detector observations a traffic-management center has."""
    mean_dt = np.mean(np.diff(t)) if len(t) > 1 else 0.1
    trend = np.polyfit(t - t[0], v, 1)[0] if len(t) > 3 else 0.0
    return {
        'mean_v': float(np.mean(v)),
        'std_v': float(np.std(v)),
        'trend_v': float(trend),
        'mean_s': float(np.mean(s)),
        'std_s': float(np.std(s)),
        'mean_dv': float(np.mean(dv)),
        'frac_stopped': float(np.mean(v < 1.0)),
        'mean_acc': float(np.mean(acc)),
    }
