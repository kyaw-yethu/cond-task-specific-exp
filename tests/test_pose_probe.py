"""Tests for the ego-motion probe's target and integration maths.

The load-bearing one is ``test_integration_inverts_to_ego_frame``: the probe
predicts per-step increments and the metrics consume a frame-0-relative track,
so if composing the increments does not reproduce ``to_ego_frame``, every
trajectory number is quietly wrong in a way no training curve would reveal.
The geometry therefore lives in the numpy-only module and is tested without
torch; the model itself is tested where torch exists.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cond_eval.traj_metrics import (increments_from_labels, integrate_increments,
                                    to_ego_frame)

# Tolerance is 1e-5, not 1e-9: increments_from_labels returns float32 because it
# feeds torch as training targets, so the round trip is exact only to about
# 6e-8 m. That is five orders of magnitude below any trajectory error worth
# reporting, and the probe's own floor will be in centimetres at best.
ROUNDTRIP_ABS = 1e-5


def _drive(n=16, seed=0, turning=True):
    """A world-frame ego track with a genuine heading change, so the rotation
    terms are exercised rather than cancelling."""
    rng = np.random.default_rng(seed)
    dt, speed = 0.1, 8.0
    yaw_rate = 0.35 if turning else 0.0
    h = np.cumsum(np.full(n, yaw_rate * dt)) + rng.uniform(-np.pi, np.pi)
    p = np.zeros((n, 2))
    for t in range(1, n):
        p[t] = p[t - 1] + speed * dt * np.array([np.cos(h[t - 1]), np.sin(h[t - 1])])
    p += rng.normal(0, 30, 2)              # arbitrary world offset
    return p[None], h[None]


# ------------------------------------------------------------------ the pivot
@pytest.mark.parametrize("turning", [True, False])
def test_integration_inverts_to_ego_frame(turning):
    """increments_from_labels -> integrate_increments must equal to_ego_frame."""
    pos, hdg = _drive(turning=turning)
    xy, heading = integrate_increments(increments_from_labels(pos, hdg))
    ref_xy, ref_h = to_ego_frame(pos, hdg)
    assert xy == pytest.approx(ref_xy, abs=ROUNDTRIP_ABS)
    assert heading == pytest.approx(ref_h, abs=ROUNDTRIP_ABS)


def test_round_trip_holds_for_a_batch_of_mixed_drives():
    pos = np.concatenate([_drive(seed=s, turning=s % 2 == 0)[0] for s in range(6)])
    hdg = np.concatenate([_drive(seed=s, turning=s % 2 == 0)[1] for s in range(6)])
    xy, heading = integrate_increments(increments_from_labels(pos, hdg))
    ref_xy, ref_h = to_ego_frame(pos, hdg)
    assert xy == pytest.approx(ref_xy, abs=ROUNDTRIP_ABS)
    assert heading == pytest.approx(ref_h, abs=ROUNDTRIP_ABS)


# ------------------------------------------------------- increment semantics
def test_increments_are_in_the_earlier_frames_body_frame():
    """Driving straight at any world heading gives pure forward increments."""
    for theta in (0.0, 1.1, -2.7, np.pi):
        n = 8
        h = np.full((1, n), theta)
        step = np.array([np.cos(theta), np.sin(theta)]) * 0.8
        p = (np.arange(n)[:, None] * step)[None] + np.array([5.0, -3.0])
        inc = increments_from_labels(p, h)
        assert inc[0, :, 0] == pytest.approx(0.8, abs=1e-6)     # forward
        assert inc[0, :, 1] == pytest.approx(0.0, abs=1e-6)     # no lateral
        assert inc[0, :, 2] == pytest.approx(0.0, abs=1e-6)     # no turn


def test_increments_are_invariant_to_world_frame():
    """The same manoeuvre at different world headings gives one increment set.
    Without this the probe would have to learn a compass it cannot observe."""
    ref = None
    for theta0 in (0.0, 0.9, -1.8):
        _, hdg = _drive(seed=1)
        hdg = hdg - hdg[:, :1] + theta0
        p = np.zeros((1, hdg.shape[1], 2))
        for t in range(1, hdg.shape[1]):
            p[:, t] = p[:, t - 1] + 0.8 * np.stack(
                [np.cos(hdg[:, t - 1]), np.sin(hdg[:, t - 1])], -1)
        inc = increments_from_labels(p, hdg)
        if ref is None:
            ref = inc
        else:
            assert inc == pytest.approx(ref, abs=1e-6)


def test_heading_increment_wraps():
    h = np.array([[np.pi - 0.05, -np.pi + 0.05]])
    p = np.zeros((1, 2, 2))
    assert increments_from_labels(p, h)[0, 0, 2] == pytest.approx(0.1, abs=1e-6)


# ------------------------------------------------- splicing from a true pose
def test_integration_can_start_from_a_given_pose():
    """What scoring a generated clip relies on: begin at the true pose at the
    cutoff so no error accumulates out of the real context frames."""
    pos, hdg = _drive(seed=3)
    inc = increments_from_labels(pos, hdg)
    full_xy, full_h = integrate_increments(inc)

    k = 5
    tail_xy, tail_h = integrate_increments(inc[:, k - 1:], full_xy[:, k - 1], full_h[:, k - 1])
    assert tail_xy[:, 0] == pytest.approx(full_xy[:, k - 1], abs=1e-9)
    assert tail_xy[:, -1] == pytest.approx(full_xy[:, -1], abs=1e-9)
    assert tail_h[:, -1] == pytest.approx(full_h[:, -1], abs=1e-9)


def test_zero_increments_stay_put():
    xy, h = integrate_increments(np.zeros((2, 15, 3)))
    assert xy == pytest.approx(np.zeros((2, 16, 2)))
    assert h == pytest.approx(np.zeros((2, 16)))


# -------------------------------------------------------------- the model
def test_probe_shapes_and_zero_init_head():
    torch = pytest.importorskip("torch")
    from cond_eval.pose_probe import PoseProbe

    model = PoseProbe(width=32).eval()
    out = model(torch.rand(2, 3, 16, 96, 96))
    assert out.shape == (2, 15, 3)
    # the head is zero-initialised, so a fresh probe predicts the training mean
    assert out.abs().max().item() == pytest.approx(0.0)


def test_probe_torch_integrate_matches_numpy():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_probe import torch_integrate

    inc = np.asarray(increments_from_labels(*_drive(seed=5)), dtype=np.float64)
    ref, _ = integrate_increments(inc)
    got = torch_integrate(torch.as_tensor(inc)).numpy()
    assert got == pytest.approx(ref[:, 1:], abs=1e-9)
