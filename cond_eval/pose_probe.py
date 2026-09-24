"""Ego-motion probe: read the depicted ego trajectory out of a 16-frame clip.

This is the extractor the trajectory metrics sit on at stage 1. DrivingGen
recovers pose with SIFT/RANSAC, UniDepthV2 and ORB-SLAM2 because it scores
arbitrary models on unlabelled footage; over 16 frames at 96x96 that pipeline
would fall through to its own constant-velocity recovery almost everywhere. Here
every clip is labelled with exact ego pose, and the camera is rigidly mounted at
a known height and pitch over a flat road, which fixes the ground-plane
homography and so makes absolute metric scale observable. A supervised regressor
is therefore both available and more accurate. It is also valid only for this
render configuration, and must be retrained if the camera or resolution changes.

**Per-step, then integrated.** The network predicts, for each adjacent frame
pair, the increment (d_forward, d_lateral, d_heading) expressed in the earlier
frame's body frame, and those 15 increments are composed as SE(2) transforms.
Three reasons this beats regressing 16 absolute positions: each step is the same
problem regardless of t, so the 15 steps share supervision instead of one clip
giving one example; the network never has to learn to integrate; and integration
can start from the true pose at the conditioning cutoff, so when scoring a
generated clip every measured increment comes from generated frames alone and no
error accumulates out of the real context.

The geometry itself lives in ``cond_eval.traj_metrics`` rather than here, so it
stays pure numpy and its round-trip against ``to_ego_frame`` can be tested
without torch. That test is the one that matters: if composing the increments
does not reproduce ``to_ego_frame``, every trajectory number is quietly wrong in
a way no training curve would reveal.

**Independence.** The trunk is a plain CNN trained from scratch on raw frames. It
shares no weights with the VAE or with any of the four candidate representations,
so nothing about the comparison is circular.

**The confound to watch.** The probe trains on real frames and runs on blurry,
checkerboarded ones. If its accuracy tracks appearance quality, the trajectory
metric becomes partly a blur detector. ``scripts/train_probe.py`` measures that
directly, reporting error on VAE round-trip reconstructions beside error on real
clips, and can train through the degradation with ``--vae-aug``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .traj_metrics import increments_from_labels, integrate_increments

__all__ = ["PoseProbe", "load_pose_probe", "build_pose_fn",
           "increments_from_labels", "integrate_increments"]


def _block(cin, cout, k, s):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, padding=k // 2, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class PoseProbe(nn.Module):
    """16 frames in, 15 pose increments out. About 1.4M params at width 256.

    Adjacent frames are stacked as 6 channels and run through one weight-shared
    2D trunk, so the per-pair problem is solved once rather than 15 times. Two
    temporal convolutions then let neighbouring pairs inform each other, which
    mainly buys noise rejection on low-texture frames.

    Increments are predicted in standardised units. ``inc_mean``/``inc_std`` are
    buffers, so a loaded checkpoint denormalises without needing the dataset.
    The head is zero-initialised, so a fresh probe predicts exactly the training
    mean rather than noise.
    """

    def __init__(self, width=256, pixel_mean=None, pixel_std=None):
        super().__init__()
        self.trunk = nn.Sequential(
            _block(6, 32, 7, 2),         # 96 -> 48
            _block(32, 64, 5, 2),        # 48 -> 24
            _block(64, 128, 3, 2),       # 24 -> 12
            _block(128, width, 3, 2),    # 12 -> 6
            _block(width, width, 3, 2),  # 6 -> 3
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv1d(width, width, 3, padding=1), nn.SiLU(inplace=True),
        )
        self.head = nn.Conv1d(width, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1) if pixel_mean is None
                             else torch.as_tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1) if pixel_std is None
                             else torch.as_tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("inc_mean", torch.zeros(3))
        self.register_buffer("inc_std", torch.ones(3))

    def forward(self, videos):
        """``videos`` (B,3,T,H,W) in [0,1] -> standardised increments (B,T-1,3)."""
        B, C, T, H, W = videos.shape
        x = videos.permute(0, 2, 1, 3, 4)                       # (B,T,3,H,W)
        x = (x - self.pixel_mean.unsqueeze(0)) / self.pixel_std.unsqueeze(0)
        pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=2)         # (B,T-1,6,H,W)
        f = self.trunk(pairs.reshape(B * (T - 1), 6, H, W))
        f = f.mean(dim=(-2, -1)).reshape(B, T - 1, -1).transpose(1, 2)
        return self.head(self.temporal(f)).transpose(1, 2)      # (B,T-1,3)

    @torch.no_grad()
    def predict_increments(self, videos):
        """Denormalised increments in metres and radians, as numpy (B,T-1,3)."""
        out = self(videos) * self.inc_std + self.inc_mean
        return out.float().cpu().numpy()


def load_pose_probe(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PoseProbe(width=ck.get("width", 256))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def build_pose_fn(device, ckpt_path=None, start_from_truth=True):
    """Factory for ``scripts/eval_dit.py --pose-module``.

    Returns ``pose_fn(videos, gt_xy=None, gt_heading=None, cutoff=None)`` giving
    ``(xy, heading)`` in the frame-0 body frame. When ``start_from_truth`` and the
    ground truth is supplied, integration begins at the true pose at ``cutoff``,
    so only increments over generated frames contribute and the result is not
    contaminated by the real context frames.
    """
    if ckpt_path is None:
        raise ValueError("build_pose_fn needs a checkpoint path (--pose-ckpt)")
    model, _ = load_pose_probe(ckpt_path, device)

    def pose_fn(videos, gt_xy=None, gt_heading=None, cutoff=None):
        inc = model.predict_increments(videos)
        if not (start_from_truth and gt_xy is not None and cutoff):
            return integrate_increments(inc)
        gt_xy = np.asarray(gt_xy, dtype=np.float64)
        gt_h = np.asarray(gt_heading, dtype=np.float64)
        k = int(cutoff)
        tail, th_tail = integrate_increments(inc[:, k - 1:], gt_xy[:, k - 1], gt_h[:, k - 1])
        xy = np.concatenate([gt_xy[:, :k - 1], tail], axis=1)
        heading = np.concatenate([gt_h[:, :k - 1], th_tail], axis=1)
        return xy, heading

    return pose_fn
