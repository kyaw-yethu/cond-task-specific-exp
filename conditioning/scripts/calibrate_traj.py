#!/usr/bin/env python
"""Run the trajectory metrics on the test split's REAL ego motion.

No model and no probe: this reads ``ego_position`` and ``ego_heading`` straight
out of ``annotations.npz`` and scores them. Two things it establishes before any
generated video is scored:

1. **What the reference-free scores read on real driving** at this clip length.
   Comfort, curvature and the composite are unitless and bounded in (0, 1], so
   "0.7" means nothing until the real-data value is known. A generated clip
   scoring *above* real driving is a warning, not a triumph: it usually means
   the motion is too smooth, not that it is better.
2. **How much of the split is scorable at all.** Comfort and curvature return
   NaN for a clip whose peak speed never reaches ``v_static`` (0.1 m/s), and
   comfort additionally needs the path to exceed ``length_eps`` (1 m). A
   dashcam corpus contains clips waiting at lights, so the scorable fraction
   is a property of the data that has to be known before differences between
   conditions are read.

Usage:
    python conditioning/scripts/calibrate_traj.py --dataset-dir <.../driving_wp_f16/test>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conditioning.lib.traj_metrics import (comfort_score, curvature_score, motion_score,
                                    quality_composite, savgol_smooth, to_ego_frame)

DT = 0.1  # the Waymo clips are 10 Hz, which is also DrivingGen's default


def describe(name, v, fmt="{:.4f}"):
    v = np.asarray(v, dtype=float)
    ok = v[~np.isnan(v)]
    nan_pct = 100.0 * (len(v) - len(ok)) / max(len(v), 1)
    if len(ok) == 0:
        print(f"  {name:22s} all NaN")
        return
    q = np.percentile(ok, [5, 25, 50, 75, 95])
    print(f"  {name:22s} mean {fmt.format(ok.mean())}  sd {fmt.format(ok.std())}  "
          f"p5 {fmt.format(q[0])}  p50 {fmt.format(q[2])}  p95 {fmt.format(q[4])}  "
          f"NaN {nan_pct:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default=None, help="write the summary here as JSON")
    ap.add_argument("--dt", type=float, default=DT)
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    meta = json.loads((d / "meta.json").read_text())
    ann = np.load(d / "annotations.npz")
    pos, hdg = ann["ego_position"], ann["ego_heading"]
    print(f"{d}")
    print(f"  {meta['num_samples']} clips, {meta['frames']} frames, {meta['img_size']} px, "
          f"channel_order {meta.get('channel_order')}")
    print(f"  ego_position {pos.shape}  ego_heading {hdg.shape}  dt={args.dt}\n")

    xy, heading = to_ego_frame(pos, hdg)
    xy_s = savgol_smooth(xy, dt=args.dt)

    step = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    length = step.sum(axis=1)
    speed = step / args.dt
    peak, avg = speed.max(axis=1), speed.mean(axis=1)

    print("=== the drive itself, from the labels ===")
    describe("path length (m)", length, "{:.2f}")
    describe("mean speed (m/s)", avg, "{:.2f}")
    describe("peak speed (m/s)", peak, "{:.2f}")
    describe("net heading change (deg)", np.degrees(np.abs(heading[:, -1])), "{:.2f}")

    below_v = float((peak < 0.1).mean())
    below_len = float((length <= 1.0).mean())
    print(f"\n  peak speed < v_static (0.1 m/s): {100 * below_v:.2f}% of clips "
          f"-> NaN for comfort and curvature")
    print(f"  path length <= length_eps (1 m):  {100 * below_len:.2f}% of clips "
          f"-> NaN for comfort")

    print("\n=== reference-free scores on REAL motion (smoothed, as upstream does) ===")
    scores = {
        "comfort_score": comfort_score(xy_s, dt=args.dt),
        "curvature_score": curvature_score(xy_s, dt=args.dt),
        "motion_score": motion_score(xy_s, dt=args.dt),
        "quality_composite": quality_composite(xy_s, dt=args.dt),
    }
    for k, v in scores.items():
        describe(k, v)

    print("\n=== the same, UNsmoothed, to size what smoothing is worth ===")
    for k, fn in (("comfort_score", comfort_score), ("curvature_score", curvature_score)):
        describe(k + " (raw)", fn(xy, dt=args.dt))

    summary = {
        "dataset_dir": str(d), "num_clips": int(len(xy)), "dt": args.dt,
        "frac_below_v_static": below_v, "frac_below_length_eps": below_len,
        "path_length_m_mean": float(length.mean()),
        "mean_speed_mps_mean": float(avg.mean()),
        **{f"real_{k}": float(np.nanmean(v)) for k, v in scores.items()},
        **{f"real_{k}_sd": float(np.nanstd(v)) for k, v in scores.items()},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
