"""Tests for the pieces ``cond_eval.dit_eval`` adds on top of the original.

``dit_eval`` imports torch and ``f_toy`` at module scope, so these skip wherever
the training environment is absent. ``run_dit_eval`` itself needs a VAE, a DiT
and I3D weights and is exercised by an actual eval run, not from here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dit_eval = pytest.importorskip(
    "cond_eval.dit_eval",
    reason="needs torch and Task_specific_JDM's f_toy on sys.path")


@pytest.mark.parametrize("k, expected", [
    (0, 0),    # unconditional: nothing is ground truth
    (1, 1),    # latent frame 0 covers raw frame 0 alone
    (2, 5),    # plus raw 1..4
    (3, 9),
])
def test_condition_raw_frame_cutoff(k, expected):
    """The predicted range every ``*_pred_range`` metric slices to. Latent frame
    0 has receptive field raw frame 0 alone; each later latent frame pools 4."""
    assert dit_eval.condition_raw_frame_cutoff(k, 4) == expected


def test_gt_pose_returns_none_when_annotations_lack_ego_fields():
    assert dit_eval._gt_pose({"lead_distance": np.zeros(16)}) == (None, None)


def test_gt_pose_matches_to_ego_frame_on_arrays():
    from cond_eval.traj_metrics import to_ego_frame

    rng = np.random.default_rng(0)
    labels = {"ego_position": rng.normal(0, 20, (3, 16, 2)),
              "ego_heading": rng.uniform(-np.pi, np.pi, (3, 16))}
    xy, h = dit_eval._gt_pose(labels)
    ref_xy, ref_h = to_ego_frame(labels["ego_position"], labels["ego_heading"])
    assert xy == pytest.approx(ref_xy)
    assert h == pytest.approx(ref_h)


def test_gt_pose_accepts_tensors():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(1)
    pos, hdg = rng.normal(0, 20, (2, 16, 2)), rng.uniform(-np.pi, np.pi, (2, 16))
    from_arrays = dit_eval._gt_pose({"ego_position": pos, "ego_heading": hdg})
    from_tensors = dit_eval._gt_pose({"ego_position": torch.from_numpy(pos),
                                      "ego_heading": torch.from_numpy(hdg)})
    assert from_tensors[0] == pytest.approx(from_arrays[0])
    assert from_tensors[1] == pytest.approx(from_arrays[1])
