"""Trajectory metrics, ported from DrivingGen's official implementation
(github.com/youngzhou1999/DrivingGen, arXiv:2601.01528, ICLR 2026).

Provenance, function by function:

| here | there |
|---|---|
| ``ade``, ``fde``            | ``drivinggen/trajs/traj_alignment.py`` |
| ``dtw_cost``                | ``traj_alignment.py``'s ``_dtw`` / ``dtw`` |
| ``comfort_score``           | ``drivinggen/trajs/traj_quality.py``'s ``comfort_score_norm`` |
| ``curvature_score``         | ``traj_quality.py``'s ``curvature_rms`` |
| ``motion_score``            | ``traj_quality.py``'s ``speed_score`` |
| ``quality_composite``       | ``traj_quality.py``'s ``get_traj_quality`` |
| ``savgol_smooth``           | ``drivinggen/z-sample_ftd.py``'s ``smooth_traj_sg`` |
| ``heading_error_deg``       | not in DrivingGen -- see below |

Numerics follow the upstream code, not the upstream docstrings, wherever the
two disagree. Places that matters:

* ``speed_score``'s docstring describes S = ln(1 + v/v_min) / ln(1 + v_max/v_min),
  but the code computes ``log1p(v_stat) / log1p(k * v_ref)``, i.e. v_min never
  enters. The code is what produced the published numbers, so the code is what
  is ported here.
* ``dtw`` returns the raw accumulated DTW cost. Its ``alpha`` parameter is
  accepted and never used. Upstream's ``__all__`` also exports ``ndtw`` and
  ``sdtw`` calls it, but no ``ndtw`` is defined anywhere in the repo, so neither
  is ported.
* ``comfort_score_norm`` takes the geometric mean of its own three sub-scores,
  while ``get_traj_quality`` takes the arithmetic mean of comfort, curvature and
  motion. Both are preserved as-is.
* Non-moving trajectories are handled inconsistently upstream: comfort and
  curvature return NaN, motion returns 0.0. Preserved, because it changes what
  ``np.nanmean`` averages over.

Heading error is ours. DrivingGen recovers pose with monocular SLAM and its
``_prep`` truncates every trajectory to its first two channels, so it carries no
heading metric at all. Here exact per-frame ``ego_heading`` is labelled, so the
error is worth reporting; it uses ``f_toy.geometry.wrap_pi``'s convention.

Conventions for every function below:

* Trajectories are ``(N, T, 2)`` or ``(T, 2)`` in metres, in the ego body frame
  of frame 0, as ``(forward, lateral)`` per ``f_toy.geometry.rotate_to_ego_frame``.
  The quality scores are invariant to that frame's handedness (curvature and yaw
  rate are both taken in absolute value), and the error metrics only require
  prediction and reference to share a frame.
* ``dt`` is 0.1 s. The Waymo clips are 10 Hz, which is also DrivingGen's default,
  so no rescaling is needed.
* Every score named ``*_score`` is higher-is-better and lies in (0, 1]. Every
  error (``ade``, ``fde``, ``dtw_cost``, ``heading_error_deg``) is
  lower-is-better.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ade", "fde", "dtw_cost", "heading_error_deg",
    "comfort_score", "curvature_score", "motion_score", "quality_composite",
    "savgol_smooth", "to_ego_frame", "trajectory_metrics",
    "increments_from_labels", "integrate_increments",
]

DT = 0.1


# --------------------------------------------------------------------------
# shape handling -- upstream ``_prep`` / ``_prep_xy``
# --------------------------------------------------------------------------
def _as_batch(arr):
    """``(T, 2)`` or ``(N, T, C)`` -> ``(N, T, 2)``, keeping only the first two
    channels. Mirrors upstream's ``arr[..., :2]`` truncation, which is how a
    3-channel reference is compared against a 2-channel prediction there."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected (T,2) or (N,T,C), got shape {a.shape}")
    return a[..., :2]


def _as_batch_pair(pred, gt):
    p, g = _as_batch(pred), _as_batch(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred and gt must share shape after truncation, got {p.shape} and {g.shape}")
    return p, g


# --------------------------------------------------------------------------
# alignment errors -- traj_alignment.py
# --------------------------------------------------------------------------
def ade(pred, gt):
    """Average displacement error per sample, metres. ``(N,)``.

    Upstream ``ade``: L2 norm at each timestep, then the mean over time."""
    p, g = _as_batch_pair(pred, gt)
    return np.linalg.norm(p - g, axis=-1).mean(axis=-1)


def fde(pred, gt):
    """Final displacement error per sample, metres. ``(N,)``."""
    p, g = _as_batch_pair(pred, gt)
    return np.linalg.norm(p[:, -1] - g[:, -1], axis=-1)


def _dtw_single(a, b):
    """Upstream ``_dtw``: classic DTW accumulated cost, L2 local distance, the
    three-predecessor recursion, no normalisation and no band constraint."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = dist + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def dtw_cost(pred, gt):
    """Raw accumulated DTW cost per sample. ``(N,)``.

    Upstream's ``dtw`` and so its ``get_dtw``. Not normalised by path length, so
    the value scales with T and is comparable only across equal-length clips.
    O(T^2) per sample, which at T=16 is negligible."""
    p, g = _as_batch_pair(pred, gt)
    return np.array([_dtw_single(p[i], g[i]) for i in range(len(p))])


def heading_error_deg(pred_heading, gt_heading):
    """Mean absolute wrapped heading error per sample, degrees. ``(N,)``.

    Not a DrivingGen metric (see module docstring). Both inputs are radians in
    the same reference, wrapped to (-pi, pi] before the absolute value so that a
    small error either side of the wrap point does not read as ~2pi."""
    p = np.atleast_2d(np.asarray(pred_heading, dtype=float))
    g = np.atleast_2d(np.asarray(gt_heading, dtype=float))
    if p.shape != g.shape:
        raise ValueError(f"heading shapes must match, got {p.shape} and {g.shape}")
    d = (p - g + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(d)).mean(axis=-1)


# --------------------------------------------------------------------------
# reference-free quality -- traj_quality.py
# --------------------------------------------------------------------------
def comfort_score(traj_xy, *, dt=DT, eps=1e-9, v_static=0.1, pct=None,
                  length_eps=1.0, j_scale=1.0, a_scale=1.0, y_scale=1.0,
                  return_components=False):
    """Upstream ``comfort_score_norm``. Per sample in (0, 1], higher is better.

    Jerk, lateral acceleration and yaw rate are aggregated over time (mean, or a
    percentile if ``pct`` is given), divided by the path length to give a
    per-metre peak, mapped through S = 1/(1 + value/scale), then combined by
    geometric mean. A trajectory that never exceeds ``v_static`` or covers at
    most ``length_eps`` metres scores NaN.

    Needs T >= 5: jerk costs four of the T samples to two rounds of central
    differencing."""
    xy = _as_batch(traj_xy)
    N, T, _ = xy.shape
    if T < 5:
        raise ValueError(f"comfort needs >= 5 frames, got {T}")

    v = (xy[:, 2:, :] - xy[:, :-2, :]) / (2 * dt)                             # (N,T-2,2)
    a = (xy[:, 2:, :] - 2 * xy[:, 1:-1, :] + xy[:, :-2, :]) / (dt ** 2)       # (N,T-2,2)
    a_c = a[:, 1:-1, :]                                                       # (N,T-4,2)
    j = (a[:, 2:, :] - a[:, :-2, :]) / (2 * dt)                               # (N,T-4,2)

    speed = np.linalg.norm(v, axis=-1)                                        # (N,T-2)
    moving = speed.max(axis=1) >= v_static

    th2 = np.arctan2(v[:, 2:, 1], v[:, 2:, 0])
    th0 = np.arctan2(v[:, :-2, 1], v[:, :-2, 0])
    yaw_rt = ((th2 - th0 + np.pi) % (2 * np.pi) - np.pi) / (2 * dt)           # (N,T-4)

    def t_reduce(x):
        return np.percentile(x, pct, axis=1) if pct is not None else x.mean(axis=1)

    jerk_p = t_reduce(np.linalg.norm(j, axis=-1))
    acc_p = t_reduce(np.linalg.norm(a_c, axis=-1))
    yaw_p = t_reduce(np.abs(yaw_rt))

    lengths = np.sum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), axis=1)    # (N,)
    valid = moving & (lengths > length_eps)

    jerk_pm = np.where(valid, jerk_p / (lengths + eps), np.nan)
    acc_pm = np.where(valid, acc_p / (lengths + eps), np.nan)
    yaw_pm = np.where(valid, yaw_p / (lengths + eps), np.nan)

    S_j = 1.0 / (1.0 + (jerk_pm / max(j_scale, eps)))
    S_a = 1.0 / (1.0 + (acc_pm / max(a_scale, eps)))
    S_y = 1.0 / (1.0 + (yaw_pm / max(y_scale, eps)))

    with np.errstate(divide="ignore", invalid="ignore"):
        comp = np.vstack([S_j, S_a, S_y])                                     # (3,N)
        score = np.exp(np.nanmean(np.log(comp), axis=0))                      # (N,)

    if return_components:
        return score, np.stack([S_j, S_a, S_y], axis=-1)
    return score


def curvature_score(traj_xy, *, dt=DT, eps=1e-9, v_static=0.1, pct=None):
    """Upstream ``curvature_rms``. Per sample in (0, 1], higher is better.

    S = 1/(1 + kappa_rms) with kappa the planar curvature
    |x' y'' - y' x''| / (x'^2 + y'^2)^1.5. With ``pct`` set, curvature above that
    percentile is dropped before the RMS. NaN when the path never moves."""
    xy = _as_batch(traj_xy)
    N, T, _ = xy.shape
    if T < 3:
        raise ValueError(f"curvature needs >= 3 frames, got {T}")

    v = np.diff(xy, axis=1) / dt                                              # (N,T-1,2)
    speed = np.linalg.norm(v, axis=-1)
    moving = speed.max(axis=1) >= v_static

    scores = np.full(N, np.nan, dtype=float)
    if moving.any():
        idx = np.where(moving)[0]
        v_mv = v[idx]                                                         # (M,T-1,2)
        a_mv = np.diff(v_mv, axis=1) / dt                                     # (M,T-2,2)

        x_dot, y_dot = v_mv[:, 1:, 0], v_mv[:, 1:, 1]                         # (M,T-2)
        x_dd, y_dd = a_mv[:, :, 0], a_mv[:, :, 1]

        num = np.abs(x_dot * y_dd - y_dot * x_dd)
        den = (x_dot ** 2 + y_dot ** 2 + eps) ** 1.5
        kappa = num / den

        if pct is not None:
            thr = np.percentile(kappa, pct, axis=-1, keepdims=True)
            kappa = np.where(kappa <= thr + eps, kappa, np.nan)

        with np.errstate(invalid="ignore"):
            rms = np.sqrt(np.nanmean(kappa ** 2, axis=-1))
        scores[idx] = 1.0 / (1.0 + rms)
    return scores


def motion_score(traj_xy, *, dt=DT, v_ref=6.0, k=2.5, v_static=0.1, use_percentile=None):
    """Upstream ``speed_score``, the term that penalises under-mobility. Per
    sample in [0, 1], higher is better.

    S = clip(log1p(v_stat) / log1p(k * v_ref), 0, 1), with v_stat the mean speed
    (or a percentile). A trajectory that never exceeds ``v_static`` scores 0.0,
    not NaN, which is how upstream distinguishes "stayed still" from "could not
    be scored".

    This is where upstream's code and docstring diverge; see the module
    docstring. ``v_static`` therefore only selects who is scored, never how."""
    xy = _as_batch(traj_xy)
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)                     # (N,T-1)

    v_stat = (v.mean(axis=1) if use_percentile is None
              else np.percentile(v, use_percentile, axis=1))
    moving = v.max(axis=1) >= v_static

    scores = np.full_like(v_stat, np.nan, dtype=float)
    denom = np.log1p(k * v_ref)
    scores[moving] = np.clip(np.log1p(v_stat[moving]) / denom, 0.0, 1.0)
    scores[~moving] = 0.0
    return scores


def quality_composite(traj_xy, **kw):
    """Upstream ``get_traj_quality``: the per-sample arithmetic mean over
    comfort, curvature and motion, NaNs skipped. ``(N,)``, higher is better.

    Note the asymmetry inherited from upstream: this mean is arithmetic, while
    ``comfort_score`` combines its own three terms geometrically."""
    comfort = comfort_score(traj_xy, **kw)
    curvature = curvature_score(traj_xy, **{k: v for k, v in kw.items() if k in
                                            ("dt", "eps", "v_static", "pct")})
    motion = motion_score(traj_xy, **{k: v for k, v in kw.items() if k == "dt"})
    stacked = np.stack([comfort, curvature, motion], axis=-1).astype(float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=-1)


# --------------------------------------------------------------------------
# preprocessing -- z-sample_ftd.py
# --------------------------------------------------------------------------
def savgol_smooth(traj_xy, *, dt=DT, win_sec=0.4, poly=3):
    """Upstream ``smooth_traj_sg``: Savitzky-Golay over the time axis, with the
    same window and order fallbacks. Batched here; upstream calls it per clip.

    This is not cosmetic. Comfort and curvature differentiate twice and three
    times respectively, so on an unsmoothed pose track they measure estimator
    noise rather than the path. Upstream smooths the prediction always, and the
    reference too whenever there is one, so both are smoothed here.

    At dt=0.1 and win_sec=0.4 the window is 5 samples with cubic order, which a
    16-frame clip accommodates."""
    from scipy.signal import savgol_filter

    xy = _as_batch(traj_xy)
    T = xy.shape[1]

    k = int(round(win_sec / dt))
    if k % 2 == 0:
        k += 1
    if k > T:
        k = T if T % 2 == 1 else T - 1
    if T == 4:
        k, poly = 3, 1
    elif poly >= k - 1:
        poly = max(1, k - 2)
    if k < 3:
        return xy

    return savgol_filter(xy, window_length=k, polyorder=poly, axis=1, mode="interp")


def to_ego_frame(positions, headings):
    """World-frame per-frame ego pose -> frame-0 ego body frame.

    ``positions`` ``(N, T, 2)`` and ``headings`` ``(N, T)`` as they sit in
    ``annotations.npz``. Returns ``(xy, heading)`` with xy in metres as
    ``(forward, lateral)`` and heading wrapped, both relative to frame 0, which
    is the reference every metric here assumes. Uses
    ``f_toy.geometry.rotate_to_ego_frame``'s convention: heading 0 faces +x,
    positive counterclockwise."""
    p = np.asarray(positions, dtype=float)
    h = np.asarray(headings, dtype=float)
    if p.ndim == 2:
        p, h = p[None], h[None]

    d = p - p[:, :1, :]
    h0 = h[:, :1]
    c, s = np.cos(h0), np.sin(h0)
    forward = d[..., 0] * c + d[..., 1] * s
    lateral = -d[..., 0] * s + d[..., 1] * c
    return np.stack([forward, lateral], axis=-1), (h - h0 + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# the one call site needs
# --------------------------------------------------------------------------
def trajectory_metrics(pred_xy, gt_xy, pred_heading=None, gt_heading=None, *,
                       dt=DT, smooth=True, pred_from=None):
    """Every metric in one pass, returned per sample so the caller can bootstrap.

    ``pred_from``, if given, additionally reports ADE and FDE restricted to
    frames ``pred_from:``, the range the DiT actually generated. The unrestricted
    figures stay the headline, matching ``dit_eval``'s FVD16 convention: the
    context frames are the same ground truth under every condition, so they move
    every condition's score alike, and the derivative-based quality scores need
    the full window to be meaningful at T=16.

    Quality scores are computed on the prediction alone, by definition
    reference-free, and on the smoothed track, as upstream does."""
    p, g = _as_batch_pair(pred_xy, gt_xy)
    if smooth:
        p, g = savgol_smooth(p, dt=dt), savgol_smooth(g, dt=dt)

    out = {
        "ade": ade(p, g),
        "fde": fde(p, g),
        "dtw": dtw_cost(p, g),
        "comfort_score": comfort_score(p, dt=dt),
        "curvature_score": curvature_score(p, dt=dt),
        "motion_score": motion_score(p, dt=dt),
        "quality_composite": quality_composite(p, dt=dt),
    }
    if pred_from is not None:
        out["ade_pred_range"] = ade(p[:, pred_from:], g[:, pred_from:])
        out["fde_pred_range"] = fde(p[:, pred_from:], g[:, pred_from:])
    if pred_heading is not None and gt_heading is not None:
        out["heading_error_deg"] = heading_error_deg(pred_heading, gt_heading)
        if pred_from is not None:
            out["heading_error_deg_pred_range"] = heading_error_deg(
                np.atleast_2d(pred_heading)[:, pred_from:],
                np.atleast_2d(gt_heading)[:, pred_from:])
    return out


# --------------------------------------------------------------------------
# per-step ego-pose increments, the parameterisation the probe predicts
# (cond_eval.pose_probe). Pure geometry, inverse of to_ego_frame above.
# --------------------------------------------------------------------------
def increments_from_labels(positions, headings):
    """World ego pose -> per-step increments in the earlier frame's body frame.

    ``positions`` (N,T,2) and ``headings`` (N,T) as they sit in
    ``annotations.npz``. Returns (N,T-1,3) of
    (d_forward, d_lateral, d_heading), heading in radians, using
    ``f_toy.geometry.rotate_to_ego_frame``'s convention: heading 0 faces +x,
    positive counterclockwise.

    float32, because these are torch training targets. The round trip through
    ``integrate_increments`` is therefore exact to about 6e-8 m, five orders of
    magnitude below any trajectory error worth reporting."""
    p = np.asarray(positions, dtype=np.float64)
    h = np.asarray(headings, dtype=np.float64)
    if p.ndim == 2:
        p, h = p[None], h[None]

    d = p[:, 1:, :] - p[:, :-1, :]                      # (N,T-1,2) world
    c, s = np.cos(h[:, :-1]), np.sin(h[:, :-1])         # the earlier frame's heading
    fwd = d[..., 0] * c + d[..., 1] * s
    lat = -d[..., 0] * s + d[..., 1] * c
    dth = (h[:, 1:] - h[:, :-1] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([fwd, lat, dth], axis=-1).astype(np.float32)


def integrate_increments(inc, start_xy=None, start_heading=None):
    """Increments -> a frame-0-relative pose track. Inverse of the above.

    ``inc`` (N,T-1,3). Returns ``(xy, heading)`` of (N,T,2) and (N,T), xy in
    metres as (forward, lateral) in frame 0's body frame. With ``start_xy`` and
    ``start_heading`` given, integration begins there instead of the origin,
    which is how a generated clip is scored from the true pose at the cutoff so
    that no error accumulates out of the real context frames.

    Heading accumulates unwrapped and is wrapped only on return, so a track that
    turns through pi does not acquire a discontinuity mid-integration."""
    a = np.asarray(inc, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    N, K, _ = a.shape

    th = np.zeros((N, K + 1))
    th[:, 1:] = np.cumsum(a[..., 2], axis=1)
    if start_heading is not None:
        th = th + np.asarray(start_heading, dtype=np.float64).reshape(N, 1)

    c, s = np.cos(th[:, :-1]), np.sin(th[:, :-1])
    step = np.stack([c * a[..., 0] - s * a[..., 1],
                     s * a[..., 0] + c * a[..., 1]], axis=-1)     # body -> frame 0
    xy = np.zeros((N, K + 1, 2))
    xy[:, 1:, :] = np.cumsum(step, axis=1)
    if start_xy is not None:
        xy = xy + np.asarray(start_xy, dtype=np.float64).reshape(N, 1, 2)

    return xy, (th + np.pi) % (2 * np.pi) - np.pi
