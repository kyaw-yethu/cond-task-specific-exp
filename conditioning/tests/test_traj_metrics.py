"""Tests for the DrivingGen metric port.

Three kinds of check, in increasing strength:

1. **Published goldens.** ``traj_alignment.py``'s own ``__main__`` demo prints
   ADE and FDE for a seeded input, and the values are recorded in its docstring.
   Those are reproduced here bit-exactly, which pins the port to the numbers
   upstream itself publishes.
2. **Analytic properties.** A straight constant-velocity path must score
   comfort 1.0 and curvature 1.0; a circle of radius R must give curvature
   1/(1+1/R); a pure translation must give ADE = FDE = the offset. These catch
   errors a golden value cannot, because they say what the metric *means*.
3. **Live parity against upstream**, skipped unless ``DRIVINGGEN_TRAJS`` points
   at a checkout's ``drivinggen/trajs``. Every ported function is compared
   element-wise on the same input.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conditioning.lib.traj_metrics import (ade, comfort_score, curvature_score, dtw_cost, fde,
                                    heading_error_deg, motion_score, quality_composite,
                                    savgol_smooth, to_ego_frame, trajectory_metrics)

DT = 0.1


def _upstream_demo_input():
    """Exactly ``traj_alignment.py``'s ``__main__`` block."""
    np.random.seed(0)
    B, T = 8, 101
    t = np.linspace(0, 4 * np.pi, T)
    single = np.stack([30 * np.cos(t), 30 * np.sin(t)], -1)
    gt = np.repeat(single[None, ...], B, axis=0)
    pred = gt + 0.4 * np.random.randn(B, T, 2) + 0.3 * np.roll(gt, 10, axis=1)
    return pred, gt


def _straight(n=16, speed=10.0, dt=DT):
    """Constant velocity along +forward: every derivative above the first is 0."""
    t = np.arange(n) * dt
    return np.stack([speed * t, np.zeros_like(t)], axis=-1)[None]


def _circle(n=64, radius=20.0, speed=10.0, dt=DT):
    """Constant-speed arc of exact curvature 1/radius."""
    omega = speed / radius
    th = omega * np.arange(n) * dt
    return np.stack([radius * np.sin(th), radius * (1 - np.cos(th))], axis=-1)[None]


# ---------------------------------------------------------------- 1. goldens
def test_ade_matches_upstream_published_value():
    pred, gt = _upstream_demo_input()
    assert float(np.mean(ade(pred, gt))) == 9.020381286932569


def test_fde_matches_upstream_published_value():
    pred, gt = _upstream_demo_input()
    assert float(np.mean(fde(pred, gt))) == 9.043563595143615


# ------------------------------------------------------- 2. analytic: errors
def test_ade_fde_of_pure_translation_equal_the_offset():
    gt = _straight()
    pred = gt + np.array([3.0, 0.0])
    assert ade(pred, gt) == pytest.approx([3.0])
    assert fde(pred, gt) == pytest.approx([3.0])


def test_ade_is_zero_for_an_exact_prediction():
    gt = _straight()
    assert ade(gt, gt) == pytest.approx([0.0])


def test_dtw_is_zero_for_an_exact_prediction():
    gt = _straight()
    assert dtw_cost(gt, gt) == pytest.approx([0.0])


def test_dtw_grows_with_error_across_the_path():
    gt = _straight()
    near = dtw_cost(gt + np.array([0.0, 0.5]), gt)[0]
    far = dtw_cost(gt + np.array([0.0, 2.0]), gt)[0]
    assert 0 < near < far


def test_dtw_absorbs_a_shift_along_the_path_but_ade_does_not():
    """DTW measures path *shape*, so a translation along the direction of travel
    is a pure time shift and the warp absorbs it. On a 1 m-spaced straight line
    a 2 m offset makes ``pred[i] == gt[i+2]`` and costs less than the same
    offset taken sideways, and less even than a 0.5 m offset that lands between
    samples.

    This is not a defect, it is what the metric is for, but it decides how the
    pair must be read: a condition that generates the right path at the wrong
    speed is caught by ADE and invisible to DTW. Report both."""
    gt = _straight()
    along = gt + np.array([2.0, 0.0])
    across = gt + np.array([0.0, 2.0])

    assert ade(along, gt) == pytest.approx(ade(across, gt))     # identical displacement
    assert dtw_cost(along, gt)[0] < dtw_cost(across, gt)[0]     # but not identical DTW
    assert dtw_cost(along, gt)[0] < dtw_cost(gt + np.array([0.5, 0.0]), gt)[0]


def test_truncates_reference_to_two_channels():
    """Upstream's ``_prep`` does ``b = b[..., :2]``, so a 3-channel reference is
    comparable against a 2-channel prediction rather than an error."""
    gt2 = _straight()
    gt3 = np.concatenate([gt2, np.full(gt2.shape[:-1] + (1,), 99.0)], axis=-1)
    assert ade(gt2, gt3) == pytest.approx([0.0])


# ----------------------------------------------------- 2. analytic: heading
def test_heading_error_is_exact_in_degrees():
    g = np.zeros((1, 8))
    p = np.full((1, 8), np.deg2rad(5.0))
    assert heading_error_deg(p, g) == pytest.approx([5.0])


def test_heading_error_wraps_across_pi():
    """Headings 0.01 either side of pi are 0.02 rad apart, not 2pi - 0.02."""
    g = np.full((1, 8), np.pi - 0.01)
    p = np.full((1, 8), -np.pi + 0.01)
    assert heading_error_deg(p, g) == pytest.approx([np.degrees(0.02)])


# ---------------------------------------------------- 2. analytic: ego frame
def test_to_ego_frame_starts_at_the_origin():
    pos = np.array([[[10.0, 5.0], [12.0, 5.0], [14.0, 5.0]]])
    hdg = np.zeros((1, 3))
    xy, h = to_ego_frame(pos, hdg)
    assert xy[0, 0] == pytest.approx([0.0, 0.0])
    assert h[0, 0] == pytest.approx(0.0)


def test_to_ego_frame_is_invariant_to_world_heading():
    """The same manoeuvre driven at any world heading must give one ego track.
    This is the property the trajectory metrics depend on: without it, two
    identical drives in different compass directions would score differently."""
    n = 12
    t = np.arange(n) * DT
    fwd, lat = 8.0 * t, 0.5 * t ** 2
    ref = None
    for theta in (0.0, np.pi / 3, -2.1, np.pi):
        c, s = np.cos(theta), np.sin(theta)
        world = np.stack([fwd * c - lat * s, fwd * s + lat * c], axis=-1) + np.array([100.0, -40.0])
        hdg = np.full(n, theta)
        xy, _ = to_ego_frame(world[None], hdg[None])
        if ref is None:
            ref = xy
        else:
            assert xy == pytest.approx(ref, abs=1e-9)
    assert ref[0, :, 0] == pytest.approx(fwd, abs=1e-9)
    assert ref[0, :, 1] == pytest.approx(lat, abs=1e-9)


# -------------------------------------------------- 2. analytic: quality
def test_constant_velocity_scores_perfect_comfort():
    """Zero jerk, zero acceleration, zero yaw rate, so all three sub-scores are
    1.0 and their geometric mean is 1.0."""
    assert comfort_score(_straight()) == pytest.approx([1.0])


def test_straight_line_scores_perfect_curvature():
    assert curvature_score(_straight()) == pytest.approx([1.0])


def test_circle_recovers_its_own_curvature():
    """S_curv = 1/(1 + kappa_rms) and a radius-R arc has kappa = 1/R. The finite
    differences are first order, so this is approximate by construction."""
    for radius in (10.0, 20.0, 50.0):
        expected = 1.0 / (1.0 + 1.0 / radius)
        assert curvature_score(_circle(radius=radius))[0] == pytest.approx(expected, rel=0.02)


def test_curvature_needs_three_frames():
    with pytest.raises(ValueError, match=">= 3 frames"):
        curvature_score(_straight(n=2))


def test_comfort_needs_five_frames():
    with pytest.raises(ValueError, match=">= 5 frames"):
        comfort_score(_straight(n=4))


def test_motion_score_rises_with_speed():
    scores = [motion_score(_straight(speed=v))[0] for v in (1.0, 5.0, 12.0)]
    assert scores == sorted(scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_motion_score_of_a_stationary_clip_is_zero_not_nan():
    """Upstream sets ``scores[~moving] = 0.0`` while comfort and curvature return
    NaN for the same clip. The asymmetry is deliberate upstream and load-bearing
    here, because it decides what ``nanmean`` averages over."""
    still = np.zeros((1, 16, 2))
    assert motion_score(still) == pytest.approx([0.0])
    assert np.isnan(comfort_score(still)).all()
    assert np.isnan(curvature_score(still)).all()


def test_composite_of_a_stationary_clip_is_zero():
    """nanmean([nan, nan, 0.0]) == 0.0, following from the asymmetry above."""
    assert quality_composite(np.zeros((1, 16, 2))) == pytest.approx([0.0])


def test_composite_is_the_arithmetic_mean_of_its_three_terms():
    """Upstream's ``get_traj_quality`` averages arithmetically, even though
    ``comfort_score_norm`` combines its own terms geometrically."""
    xy = _circle(n=16, radius=30.0)
    expected = np.nanmean([comfort_score(xy)[0], curvature_score(xy)[0], motion_score(xy)[0]])
    assert quality_composite(xy)[0] == pytest.approx(expected)


def test_comfort_penalises_jerk():
    smooth = _circle(n=32, radius=40.0)
    jerky = smooth.copy()
    jerky[0, ::2] += np.array([0.0, 0.25])       # alternating lateral kick
    assert comfort_score(jerky)[0] < comfort_score(smooth)[0]


# ------------------------------------------------------ 2. analytic: smoothing
def test_savgol_preserves_shape_and_reduces_noise():
    rng = np.random.default_rng(0)
    clean = _circle(n=16, radius=25.0)
    noisy = clean + rng.normal(0, 0.05, clean.shape)
    out = savgol_smooth(noisy)
    assert out.shape == clean.shape
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


def test_savgol_leaves_a_cubic_path_alone():
    """Window 5 with polyorder 3 reproduces any cubic exactly, interior and
    edges, since ``mode='interp'`` fits the edge polynomial rather than padding."""
    t = np.arange(16) * DT
    cubic = np.stack([2.0 * t ** 3 - t, 0.5 * t ** 2], axis=-1)[None]
    assert savgol_smooth(cubic) == pytest.approx(cubic, abs=1e-9)


# ------------------------------------------------------- 2. the call wrapper
def test_trajectory_metrics_reports_every_metric_and_the_restricted_range():
    gt = _circle(n=16, radius=30.0)
    pred = gt + np.array([0.4, -0.2])
    out = trajectory_metrics(pred, gt, pred_heading=np.zeros((1, 16)),
                             gt_heading=np.zeros((1, 16)), pred_from=5)
    for key in ("ade", "fde", "dtw", "comfort_score", "curvature_score", "motion_score",
                "quality_composite", "ade_pred_range", "fde_pred_range",
                "heading_error_deg", "heading_error_deg_pred_range"):
        assert key in out, key
        assert np.asarray(out[key]).shape == (1,)


def test_trajectory_metrics_omits_heading_when_absent():
    gt = _circle(n=16, radius=30.0)
    out = trajectory_metrics(gt + 0.1, gt)
    assert "heading_error_deg" not in out
    assert "ade_pred_range" not in out


def test_trajectory_metrics_smoothing_is_applied_to_both_sides():
    """A prediction equal to the reference must score ADE 0 whether or not
    smoothing runs, which only holds if both sides are smoothed."""
    gt = _circle(n=16, radius=30.0) + np.random.default_rng(3).normal(0, 0.05, (1, 16, 2))
    assert trajectory_metrics(gt, gt, smooth=True)["ade"] == pytest.approx([0.0])


# ------------------------------------------------------------- 3. live parity
UPSTREAM = os.environ.get("DRIVINGGEN_TRAJS")
needs_upstream = pytest.mark.skipif(
    not UPSTREAM or not Path(UPSTREAM).is_dir(),
    reason="set DRIVINGGEN_TRAJS to a DrivingGen checkout's drivinggen/trajs to run parity tests")


def _mixed_tracks(n=32, T=16, seed=7):
    """Driving-like tracks at 10 Hz, deliberately including a stationary one so
    the NaN paths are exercised."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT
    out = []
    for i in range(n):
        speed = 0.0 if i == 0 else rng.uniform(0.5, 18.0)
        fwd = speed * t
        lat = rng.normal(0.0, 0.04) * fwd ** 2
        out.append(np.stack([fwd, lat], -1) + rng.normal(0, 0.03, (T, 2)))
    return np.array(out)


@needs_upstream
@pytest.mark.parametrize("name", ["comfort", "curvature", "motion"])
def test_quality_scores_match_upstream_elementwise(name):
    sys.path.insert(0, UPSTREAM)
    import traj_quality as up

    tracks = _mixed_tracks()
    ours, theirs = {
        "comfort": (comfort_score(tracks), up.comfort_score_norm(tracks, reduce="none")),
        "curvature": (curvature_score(tracks), up.curvature_rms(tracks, reduce="none")),
        "motion": (motion_score(tracks), up.speed_score(tracks, reduce="none")),
    }[name]
    theirs = np.asarray(theirs, dtype=float)
    assert np.array_equal(np.isnan(ours), np.isnan(theirs))
    assert ours[~np.isnan(ours)] == pytest.approx(theirs[~np.isnan(theirs)], rel=1e-12, abs=1e-12)


@needs_upstream
def test_composite_matches_upstream():
    sys.path.insert(0, UPSTREAM)
    import traj_quality as up

    tracks = _mixed_tracks()
    assert float(np.nanmean(quality_composite(tracks))) == pytest.approx(
        up.get_traj_quality(tracks), rel=1e-12, abs=1e-12)


@needs_upstream
def test_ade_and_dtw_match_upstream_entry_points():
    sys.path.insert(0, UPSTREAM)
    import traj_alignment as up

    pred, gt = _upstream_demo_input()
    pred, gt = pred[:, :16], gt[:, :16]
    assert float(np.nanmean(ade(pred, gt))) == pytest.approx(up.get_ade(pred, gt), rel=1e-12)
    assert float(np.nanmean(dtw_cost(pred, gt))) == pytest.approx(up.get_dtw(pred, gt), rel=1e-12)
