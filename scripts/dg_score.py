"""Score the extracted trajectories against nuScenes ground truth.

This is the calibration: every clip is real footage, so whatever error comes out
is the benchmark's own noise floor rather than a property of any generator.

Four cuts are reported side by side because each isolates a different source of
that floor:

reference     ego origin (what their pipeline compares against) or the camera
              optical centre, which is where the SLAM actually estimates motion.
              The two differ by the sensor extrinsic, about 1.7 m forward, and
              that offset rotates with heading.
intrinsics    predicted by UniDepth, their default, against the real CAM_FRONT
              calibration.
with_scale    their alignment fixes scale at 1, so ADE carries UniDepth's
              absolute-scale error. Allowing scale separates it from path shape.
failure rate  the fraction of frame pairs whose pose estimate failed and was
              replaced by the random-yaw fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cond_eval.dg_eval import score as S


def load(clips_dir: str, traj_dir: str):
    names, locs, gt_ego, gt_cam, fails, meta = [], [], [], [], [], []
    for f in sorted(os.listdir(traj_dir)):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        mpath = os.path.join(clips_dir, name, "meta.json")
        if not os.path.exists(mpath):
            continue
        d = np.load(os.path.join(traj_dir, f), allow_pickle=False)
        with open(mpath) as fh:
            m = json.load(fh)
        names.append(name)
        locs.append(d["locs"])
        gt_ego.append(np.asarray(m["ego_xy"], float))
        gt_cam.append(np.asarray(m["cam_xy"], float))
        fails.append(json.loads(str(d["meta"])))
        meta.append(m)
    return names, locs, gt_ego, gt_cam, fails, meta


def run_cut(repo, locs, gts, *, with_scale, dt):
    preds, refs, scales = [], [], []
    for p, g in zip(locs, gts):
        pr, gr, s = S.prepare_pair(repo, p, g, with_scale=with_scale, dt=dt)
        preds.append(pr); refs.append(gr); scales.append(s)
    preds = np.stack(preds); refs = np.stack(refs)
    per = S.score_pairs(repo, preds, refs, dt=dt)
    out = S.summarise(per)
    out["_scale"] = {"median": float(np.median(scales)),
                     "p05": float(np.percentile(scales, 5)),
                     "p95": float(np.percentile(scales, 95))}
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--traj", default="/root/driving-gen/traj/dg")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--out", default="out/dg_calibration")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--tag", default="dg")
    ap.add_argument("--modes", default="",
                    help="comma-separated subset; default is every mode present")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {"tag": args.tag, "dt": args.dt, "modes": {}}

    wanted = [m for m in args.modes.split(",") if m] or sorted(os.listdir(args.traj))
    for mode in wanted:
        traj_dir = os.path.join(args.traj, mode)
        if not os.path.isdir(traj_dir):
            continue
        names, locs, gt_ego, gt_cam, fails, meta = load(args.clips, traj_dir)
        if not names:
            print("no results in", traj_dir)
            continue

        fr = np.array([f["fail_rate"] for f in fails])
        resid = np.concatenate([np.asarray(m["grid_residual_ms"]) for m in meta])
        block = {
            "n_clips": len(names),
            "fail_rate": {"mean": float(fr.mean()), "median": float(np.median(fr)),
                          "p95": float(np.percentile(fr, 95)), "max": float(fr.max()),
                          "clips_all_ok": int((fr == 0).sum())},
            "grid_residual_ms": {"mean": float(resid.mean()),
                                 "p95": float(np.percentile(resid, 95)),
                                 "max": float(resid.max())},
            "cuts": {},
        }
        print("\n=== %s | %d clips | mean pose-failure rate %.3f ==="
              % (mode, len(names), fr.mean()), flush=True)

        for ref_name, gts in (("ego", gt_ego), ("camera", gt_cam)):
            for ws in (False, True):
                key = "%s/%s" % (ref_name, "scaled" if ws else "fixed_scale")
                summ, per = run_cut(args.repo, locs, gts, with_scale=ws, dt=args.dt)
                block["cuts"][key] = summ
                np.savez_compressed(
                    os.path.join(args.out, "per_clip_%s_%s_%s.npz"
                                 % (args.tag, mode, key.replace("/", "_"))),
                    names=np.asarray(names), fail_rate=fr, **per)
                print("  %-22s ADE %6.3f  FDE %6.3f  DTW %8.2f  Hausdorff %6.3f  SR %.3f"
                      % (key, summ["ade"]["mean"], summ["fde"]["mean"],
                         summ["dtw"]["mean"], summ["hausdorff"]["mean"],
                         summ["success_rate"]["mean"]), flush=True)
                if ws:
                    sc = summ["_scale"]
                    print("      alignment scale: median %.3f  [p05 %.3f, p95 %.3f]"
                          % (sc["median"], sc["p05"], sc["p95"]))

        report["modes"][mode] = block

    path = os.path.join(args.out, "report_%s.json" % args.tag)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
