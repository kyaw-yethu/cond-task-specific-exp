"""Why the tail clips fail.

The extractor composes 100 relative motions, so a single bad pair rotates
everything after it. This pulls the per-pair diagnostics the extractor recorded
alongside each clip, correlates them against the clip's error, and localises the
step where a failing clip goes wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image

from dg_eval.lib import score as S


def clip_facts(clips, traj, name, repo):
    with open(os.path.join(clips, name, "meta.json")) as f:
        m = json.load(f)
    d = np.load(os.path.join(traj, name + ".npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))

    gt = np.asarray(m["ego_xy"], float)
    t = np.asarray(m["t_us"], float) / 1e6
    step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    dt = np.diff(t)
    speed = step / np.maximum(dt, 1e-6)
    yaw = np.unwrap(np.asarray(m["ego_yaw"], float))

    img = np.asarray(Image.open(
        os.path.join(clips, name, "images",
                     sorted(os.listdir(os.path.join(clips, name, "images")))[len(os.listdir(os.path.join(clips, name, "images"))) // 2])).convert("L"))

    fixed, ref, _ = S.prepare_pair(repo, d["locs"], gt, with_scale=False)
    dg = S.load_drivinggen(repo)
    ade = float(dg.align.ade(fixed[None], ref[None]).ravel()[0])

    # per-step error growth, in the aligned frame
    err = np.linalg.norm(fixed - ref, axis=1)

    return {
        "name": name,
        "ade": ade,
        "driven": float(step.sum()),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "yaw_total": float(np.abs(np.diff(yaw)).sum() * 180 / np.pi),
        "brightness": float(img.mean()),
        "fail_rate": float(meta["fail_rate"]),
        "depth_median": float(meta["depth_median"]),
        "matches": float(d["n_matches"].mean()),
        "pnp_pts": float(d["n_matched_points"].mean()),
        "pnp_min": float(d["n_matched_points"].min()),
        "err": err,
        "ok": d["ok"],
        "pts_per_pair": d["n_matched_points"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--traj", default="/root/driving-gen/traj/dg/predicted")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--percl", default="out/dg_calibration/"
                                       "per_clip_dg_predicted_ego_fixed_scale.npz")
    args = ap.parse_args()

    names = [str(x) for x in np.load(args.percl, allow_pickle=True)["names"]]
    rows = [clip_facts(args.clips, args.traj, n, args.repo) for n in names]

    driven = np.array([r["driven"] for r in rows])
    ade = np.array([r["ade"] for r in rows])
    moving = driven >= 10
    R = [r for r, m in zip(rows, moving) if m]
    a = ade[moving]

    print("moving clips: %d\n" % len(R))
    print("%-14s %8s %8s   %s" % ("factor", "rho", "rho|log", "bottom vs top quartile"))
    keys = ["speed_mean", "speed_max", "driven", "yaw_total", "brightness",
            "fail_rate", "depth_median", "matches", "pnp_pts", "pnp_min"]
    la = np.log10(np.maximum(a, 1e-3))
    for k in keys:
        v = np.array([r[k] for r in R])
        rho = np.corrcoef(v, a)[0, 1]
        rho_l = np.corrcoef(v, la)[0, 1]
        lo = v[a <= np.percentile(a, 25)].mean()
        hi = v[a >= np.percentile(a, 75)].mean()
        print("%-14s %8.3f %8.3f   best %8.2f   worst %8.2f" % (k, rho, rho_l, lo, hi))

    print("\n=== the three worst clips, step by step ===")
    worst = [R[i]["name"] for i in np.argsort([r["ade"] for r in R])[::-1][:3]]
    for name in worst:
        r = next(x for x in rows if x["name"] == name)
        err, ok, pts = r["err"], r["ok"], r["pts_per_pair"]
        jump = np.diff(err)
        worst = int(np.argmax(jump))
        print("\n%s  ADE %.2f  driven %.0f m  mean speed %.1f m/s  yaw %.0f deg  "
              "brightness %.0f" % (name, r["ade"], r["driven"], r["speed_mean"],
                                   r["yaw_total"], r["brightness"]))
        print("  failed pairs      %d of %d" % ((~ok).sum(), ok.size))
        print("  PnP points        mean %.0f  min %.0f" % (pts.mean(), pts.min()))
        q = len(err) // 4
        print("  error at frame  %3d: %5.2f  %3d: %5.2f  %3d: %5.2f  %3d: %5.2f"
              % (q, err[q], 2 * q, err[2 * q], 3 * q, err[3 * q],
                 len(err) - 1, err[-1]))
        print("  largest jump      +%.2f m at pair %d (%d PnP points, ok=%s)"
              % (jump[worst], worst, pts[worst], bool(ok[worst])))
        half = err[:len(err) // 2].max()
        print("  max error in first half: %.2f m  -> %s"
              % (half, "breaks early" if half > 0.3 * err[-1] else "breaks late"))
        bad = np.where(pts < 40)[0]
        if len(bad):
            print("  pairs under 40 PnP points: %d (first at %d)" % (len(bad), bad[0]))


if __name__ == "__main__":
    main()
