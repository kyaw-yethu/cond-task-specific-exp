#!/usr/bin/env python
"""Probe trajectories against ground truth on test clips, on real frames and on
their VAE round trip.

One row per clip: frame 0 with the three paths projected through the render
camera, the same frame after the round trip, and the three paths from above.
Clips are picked from the ground truth to cover left and right turns, fast and
straight driving, a median clip and a slow one.

    python conditioning/scripts/plot_probe_overlay.py \
        --data data/womd_7hz_f33/test --probe out/probe_womd/probe.pt \
        --vae-dir checkpoints/waymo/vae_miniwan_womd --out out/probe_womd/overlay.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "third_party/Task_specific_JDM")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from f_toy.data import get_dataset_plugin
from f_toy.evaluation.checkpoints import load_vae
from conditioning.lib.pose_probe import load_pose_probe
from conditioning.lib.traj_metrics import increments_from_labels, integrate_increments

# MetaDrive render camera of womd_7hz_f33 and driving_wp_f16 (womd/lib/render.py):
# 1.2 m ahead of the ego origin, 1.3 m up, pitched 15 degrees down, 60 degree FOV.
CAM_FWD, CAM_UP, PITCH, FOV = 1.2, 1.3, np.radians(-15.0), np.radians(60.0)
GT, REAL, RT = "#2ca02c", "#1f77b4", "#ff7f0e"


def project(xy, size):
    """(T,2) ground points, forward and left in the frame-0 body frame -> pixels."""
    d = xy[:, 0] - CAM_FWD
    left = xy[:, 1]
    up = np.full_like(d, -CAM_UP)
    zc = d * np.cos(PITCH) + up * np.sin(PITCH)
    yc = -d * np.sin(PITCH) + up * np.cos(PITCH)
    xc = -left
    f = (size / 2) / np.tan(FOV / 2)
    ok = zc > 0.5
    u = size / 2 + f * xc / np.where(ok, zc, 1)
    v = size / 2 - f * yc / np.where(ok, zc, 1)
    return u[ok], v[ok]


def pick(xy, hd):
    """Indices covering the motion range, chosen from the ground truth alone."""
    length = np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(1)
    turn = np.degrees(np.unwrap(hd, axis=1)[:, -1] - hd[:, 0])
    moving = length > 5
    idx = []
    def take(order):
        for i in order:
            if i not in idx:
                idx.append(int(i)); return
    take(np.argsort(-np.where(moving, turn, -1e9)))                      # sharpest left
    take(np.argsort(-np.where(moving & (turn > 20) & (turn < 60), turn, -1e9)))
    take(np.argsort(np.where(moving, turn, 1e9)))                        # sharpest right
    take(np.argsort(np.where(moving & (turn < -20) & (turn > -60), turn, 1e9)))
    straight = np.abs(turn) < 3
    take(np.argsort(-np.where(straight, length, -1)))                    # fastest straight
    take(np.argsort(-np.where(straight, length, -1))[50:])
    take(np.argsort(np.abs(length - np.median(length))))                 # median
    take(np.argsort(np.abs(length - 3.0)))                               # slow, about 3 m
    return idx, length, turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/womd_7hz_f33/test")
    ap.add_argument("--probe", default="out/probe_womd/probe.pt")
    ap.add_argument("--vae-dir", default="checkpoints/waymo/vae_miniwan_womd")
    ap.add_argument("--out", default="out/probe_womd/overlay.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", type=int, nargs="+", default=None,
                    help="test clip indices to plot instead of the automatic pick")
    ap.add_argument("--title", default="Ego-motion probe on womd_7hz_f33 test clips (33 frames, 7 Hz, 4.6 s)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = get_dataset_plugin({"DATASET": "waymo"}).DiskDataset(args.data)
    ann = np.load(Path(args.data) / "annotations.npz")
    gt_inc = increments_from_labels(ann["ego_position"], ann["ego_heading"])
    gxy, ghd = integrate_increments(gt_inc)
    idx, length, turn = pick(gxy, ghd)
    if args.clips:
        idx = list(args.clips)

    probe, _ = load_pose_probe(args.probe, dev)
    vae, _ = load_vae(args.vae_dir, "best", dev)
    torch.manual_seed(args.seed)
    videos = torch.stack([ds[i][0] for i in idx]).to(dev)
    with torch.no_grad():
        recon = torch.cat([vae(videos[j:j + 2])[0].clamp(0, 1) for j in range(0, len(idx), 2)])
    rxy, _ = integrate_increments(probe.predict_increments(videos))
    txy, _ = integrate_increments(probe.predict_increments(recon))

    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for r, i in enumerate(idx):
        g = gxy[i]
        ade_r = np.linalg.norm(rxy[r] - g, axis=-1)[1:].mean()
        ade_t = np.linalg.norm(txy[r] - g, axis=-1)[1:].mean()
        for c, (img, name) in enumerate(((videos[r], "real frame 0"), (recon[r], "MiniWan round trip, frame 0"))):
            ax = axes[r, c]
            size = img.shape[-1]
            ax.imshow(img[:, 0].permute(1, 2, 0).cpu().numpy(), extent=(0, size, size, 0))
            for xy, col, lw in ((g, GT, 2.2), (rxy[r], REAL, 1.6), (txy[r], RT, 1.6)):
                u, v = project(xy, size)
                ax.plot(u, v, color=col, lw=lw)
            ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.axis("off")
            ax.set_title(name, fontsize=9)
        ax = axes[r, 2]
        ax.plot(-g[:, 1], g[:, 0], color=GT, lw=2.2, label="ground truth")
        ax.plot(-rxy[r][:, 1], rxy[r][:, 0], color=REAL, lw=1.6, label=f"probe, real  ADE {ade_r:.2f} m")
        ax.plot(-txy[r][:, 1], txy[r][:, 0], color=RT, lw=1.6, label=f"probe, round trip  ADE {ade_t:.2f} m")
        ax.plot(0, 0, "k^", ms=6)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.set_xlabel("right (m)", fontsize=8); ax.set_ylabel("forward (m)", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="best")
        ax.set_title(f"test clip {i}: path {length[i]:.1f} m, turn {turn[i]:+.0f} deg", fontsize=9)
    fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)
    for r, i in enumerate(idx):
        print(f"clip {i:5d}  path {length[i]:5.1f} m  turn {turn[i]:+6.1f} deg  "
              f"ADE real {np.linalg.norm(rxy[r] - gxy[i], axis=-1)[1:].mean():.2f}  "
              f"round trip {np.linalg.norm(txy[r] - gxy[i], axis=-1)[1:].mean():.2f}")


if __name__ == "__main__":
    main()
