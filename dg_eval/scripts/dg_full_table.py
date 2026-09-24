"""Every trajectory metric, for every configuration.

The summary tables carry a subset; this prints the whole set that DrivingGen's
`trajs` package defines, plus heading error, which their metric set omits but
their saved pose chain supports. Heading is taken relative to frame 0, so the
global rotation their alignment fits drops out and no Procrustes angle is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

def cells(geom: str):
    """The five configurations, tagged for whichever geometry is being read."""
    return [
        (geom, "predicted", "top + predicted (as published)"),
        (geom, "calibrated_slam", "top + calibrated_slam"),
        (geom, "calibrated_both", "top + calibrated_both"),
        ("full_" + geom, "predicted", "full + predicted"),
        ("full_" + geom, "calibrated_both", "full + calibrated_both (repaired)"),
    ]


def traj_dirs(geom: str):
    return {geom: "/root/driving-gen/traj/%s" % geom,
            "full_" + geom: "/root/driving-gen/traj_roi/full_%s" % geom}


def yaw_of(R):
    """Yaw about the vertical axis of their camera-style world frame."""
    return np.arctan2(R[:, 0, 2], R[:, 2, 2])


def quality_metrics(repo, traj_dir, clips, names):
    """Their reference-free quality triple and the composite it averages into.

    `get_traj_quality` returns a single number over the whole set, so the three
    components are called directly to keep per-clip values.
    """
    from dg_eval.lib import score as S
    dg = S.load_drivinggen(repo)
    q = dg.quality
    preds = []
    keep = []
    for n in names:
        f = os.path.join(traj_dir, n + ".npz")
        if not os.path.exists(f):
            continue
        with open(os.path.join(clips, n, "meta.json")) as fh:
            m = json.load(fh)
        d = np.load(f, allow_pickle=True)
        p, _, _ = S.prepare_pair(repo, d["locs"], np.asarray(m["ego_xy"], float),
                                 with_scale=False)
        preds.append(p)
        keep.append(n)
    preds = np.stack(preds)
    comfort = np.asarray(q.comfort_score_norm(preds, reduce="none"), float)
    crms = np.asarray(q.curvature_rms(preds, reduce="none"), float)
    speed = np.asarray(q.speed_score(preds, reduce="none"), float)
    composite = np.nanmean(np.stack([comfort, crms, speed], -1), axis=-1)
    return comfort, crms, speed, composite


def heading_error(traj_dir, clips, names):
    out = []
    for n in names:
        f = os.path.join(traj_dir, n + ".npz")
        if not os.path.exists(f):
            out.append(np.nan); continue
        d = np.load(f, allow_pickle=True)
        pred = np.unwrap(yaw_of(d["poses_R"]))
        pred = pred - pred[0]
        with open(os.path.join(clips, n, "meta.json")) as fh:
            gt = np.unwrap(np.asarray(json.load(fh)["ego_yaw"], float))
        gt = gt - gt[0]
        k = min(len(pred), len(gt))
        err = np.abs(np.arctan2(np.sin(pred[:k] - gt[:k]), np.cos(pred[:k] - gt[:k])))
        out.append(float(np.degrees(err).mean()))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/dg_calibration")
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--geom", default="dg", choices=["dg", "stage2"])
    args = ap.parse_args()
    CELLS, TRAJ = cells(args.geom), traj_dirs(args.geom)

    ref = np.load(os.path.join(args.base, "per_clip_%s_predicted_ego_fixed_scale.npz"
                               % args.geom),
                  allow_pickle=True)
    names = [str(x) for x in ref["names"]]
    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            p = np.asarray(json.load(f)["ego_xy"], float)
        driven.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    mv = np.asarray(driven) >= 10

    keys = ["ade", "fde", "dtw", "hausdorff", "success_rate",
            "dynamic_consistency", "comfort", "traj_consistency"]
    hdr = ["ADE", "FDE", "DTW", "Hausd", "SR@3m", "DynCons", "Comfort", "TrajCons", "Head"]
    print("%-34s" % "configuration" + "".join("%9s" % h for h in hdr))
    for tag, mode, label in CELLS:
        f = os.path.join(args.base, "per_clip_%s_%s_ego_fixed_scale.npz" % (tag, mode))
        if not os.path.exists(f):
            print("%-34s  missing" % label); continue
        d = np.load(f, allow_pickle=True)
        row = []
        for k in keys:
            v = np.asarray(d[k], float)[mv]
            row.append(np.nanmean(v))
        hd = heading_error(os.path.join(TRAJ[tag], mode), args.clips,
                           [n for n, m in zip(names, mv) if m])
        row.append(np.nanmean(hd))
        print("%-34s" % label + "".join("%9.3f" % v for v in row))

    print("\nmedians, same order")
    print("%-34s" % "configuration" + "".join("%9s" % h for h in hdr))
    for tag, mode, label in CELLS:
        f = os.path.join(args.base, "per_clip_%s_%s_ego_fixed_scale.npz" % (tag, mode))
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        row = [np.nanmedian(np.asarray(d[k], float)[mv]) for k in keys]
        hd = heading_error(os.path.join(TRAJ[tag], mode), args.clips,
                           [n for n, m in zip(names, mv) if m])
        row.append(np.nanmedian(hd))
        print("%-34s" % label + "".join("%9.3f" % v for v in row))

    print("\nreference-free quality, mean over moving clips")
    print("%-34s%9s%9s%9s%11s" % ("configuration", "Comfort", "CurvRMS", "Speed",
                                  "Composite"))
    moving_names = [n for n, m in zip(names, mv) if m]
    for tag, mode, label in CELLS:
        td = os.path.join(TRAJ[tag], mode)
        if not os.path.isdir(td):
            continue
        c, cr, sp, comp = quality_metrics(args.repo, td, args.clips, moving_names)
        print("%-34s%9.3f%9.3f%9.3f%11.3f"
              % (label, np.nanmean(c), np.nanmean(cr), np.nanmean(sp),
                 np.nanmean(comp)))


if __name__ == "__main__":
    main()
