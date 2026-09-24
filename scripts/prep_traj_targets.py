#!/usr/bin/env python
"""Compact per-clip targets for Traj-VAE v2, from a womd_7hz_f33 split.

annotations.npz is 3.2 GB over 44 keys; training needs a few of them, plus one
derived target. This writes a single npz per split, small enough to hold in RAM
in every dataloader worker:

    ego_position      (N, 33, 2) float32   world frame
    ego_heading       (N, 33)    float32
    ego_speed         (N, 33)    float32
    future_position   (N, 28, 2) float32   after the clip, 4 s at 7 Hz
    future_heading    (N, 28)    float32
    future_valid      (N, 28)    float32
    occupancy         (N, 33, 12, 12) uint8  visible agents in the image plane
    lane_curvature    (N, 33)    float32
    in_intersection   (N, 33)    float32
    tl_state          (N, 33)    int8     -1 none, 0 unknown, 1 stop, 2 caution, 3 go

Occupancy is built from agents_bbox_2d and agents_visible: each visible box is
painted onto the 12x12 grid the latent shares with z, so the agent head predicts
something aligned with r spatially.

    python scripts/prep_traj_targets.py --split train
"""
import argparse
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data/womd_7hz_f33"
GRID, IMG = 12, 96


def occupancy(bbox, visible):
    """bbox (N, T, A, 4) as (cx, cy, w, h) in pixels, visible (N, T, A) -> (N, T, 12, 12) uint8."""
    N, T, A, _ = bbox.shape
    out = np.zeros((N, T, GRID, GRID), np.uint8)
    cx, cy, w, h = (bbox[..., i] for i in range(4))
    x0 = np.clip((cx - w / 2) / IMG * GRID, 0, GRID - 1e-3)
    x1 = np.clip((cx + w / 2) / IMG * GRID, 0, GRID - 1e-3)
    y0 = np.clip((cy - h / 2) / IMG * GRID, 0, GRID - 1e-3)
    y1 = np.clip((cy + h / 2) / IMG * GRID, 0, GRID - 1e-3)
    vis = visible > 0.5
    n, t, a = np.nonzero(vis)
    for i, j, k in zip(n, t, a):
        out[i, j, int(y0[i, j, k]):int(y1[i, j, k]) + 1, int(x0[i, j, k]):int(x1[i, j, k]) + 1] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = DATA / args.split
    ann = np.load(d / "annotations.npz")
    print(f"{args.split}: keys {len(ann.files)}", flush=True)

    out = {}
    out["ego_position"] = ann["ego_position"].astype(np.float32)
    out["ego_heading"] = ann["ego_heading"].astype(np.float32)
    out["ego_speed"] = ann["ego_speed"].astype(np.float32)
    out["future_position"] = ann["future_ego_position"].astype(np.float32)
    out["future_heading"] = ann["future_ego_heading"].astype(np.float32)
    out["future_valid"] = ann["future_valid"].astype(np.float32)
    for src, dst in (("lane_curvature", "lane_curvature"), ("in_intersection", "in_intersection")):
        out[dst] = ann[src].astype(np.float32)
    out["tl_state"] = ann["tl_state"].astype(np.int8)
    print("  shapes:", {k: v.shape for k, v in out.items()}, flush=True)

    bbox = ann["agents_bbox_2d"]
    vis = ann["agents_visible"]
    print(f"  occupancy from {bbox.shape} boxes, {float(vis.mean()) * 100:.1f}% visible", flush=True)
    out["occupancy"] = occupancy(bbox.astype(np.float32), vis.astype(np.float32))
    print(f"  occupancy mean {out['occupancy'].mean():.4f}", flush=True)

    p = Path(args.out) if args.out else d / "traj_targets.npz"
    np.savez(p, **out)
    print(f"wrote {p} ({p.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
