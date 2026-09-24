"""Does their top-half feature mask cost accuracy?

`drive_roi_mask` keeps `[0 : h/2]`, so features never come from the road. The
complementary slice is what the function's name suggests, and dropping the cut
altogether is the third option. Same clips, same seeds, same everything else, so
the comparison is paired and deterministic.

Two groups, because a change that rescues the tail may still hurt the clips that
already work: the 40 worst moving clips, and 40 drawn at random from the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from dg_eval.lib import score as S

GROUPS = ("tail", "control")


def pick(args):
    d = np.load(args.percl, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    ade = d["ade"]
    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            p = np.asarray(json.load(f)["ego_xy"], float)
        driven.append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
    moving = np.asarray(driven) >= 10
    idx = np.where(moving)[0]
    order = idx[np.argsort(ade[idx])[::-1]]
    tail = [names[i] for i in order[:args.n]]
    rest = [names[i] for i in order[args.n:]]
    rng = np.random.default_rng(2026)
    control = [rest[i] for i in rng.choice(len(rest), args.n, replace=False)]
    return tail, control, dict(zip(names, ade))


def score_set(args, roi, names, base_ade):
    traj = os.path.join(args.traj_root, roi, args.intrinsics)
    got, out = [], {}
    for n in names:
        f = os.path.join(traj, n + ".npz")
        if not os.path.exists(f):
            continue
        with open(os.path.join(args.clips, n, "meta.json")) as fh:
            m = json.load(fh)
        d = np.load(f, allow_pickle=True)
        fixed, ref, _ = S.prepare_pair(args.repo, d["locs"],
                                       np.asarray(m["ego_xy"], float),
                                       with_scale=False)
        dg = S.load_drivinggen(args.repo)
        out[n] = {
            "ade": float(dg.align.ade(fixed[None], ref[None]).ravel()[0]),
            "pnp": float(d["n_matched_points"].mean()),
            "matches": float(d["n_matches"].mean()),
            "fail": float(json.loads(str(d["meta"]))["fail_rate"]),
        }
        got.append(n)
    return out, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--traj-root", default="/root/driving-gen/traj_roi")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--intrinsics", default="predicted")
    ap.add_argument("--percl", default="out/dg_calibration/"
                                       "per_clip_dg_predicted_ego_fixed_scale.npz")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--emit", default="")
    ap.add_argument("--rois", default="top,bottom,full")
    args = ap.parse_args()

    tail, control, base = pick(args)
    if args.emit:
        with open(args.emit, "w") as f:
            f.write("\n".join(tail + control) + "\n")
        print("wrote %s: %d tail + %d control" % (args.emit, len(tail), len(control)))
        return

    rois = args.rois.split(",")
    results = {r: score_set(args, r, tail + control, base)[0] for r in rois}

    for gname, group in (("tail 40 (worst moving)", tail), ("control 40 (random)", control)):
        print("\n=== %s ===" % gname)
        print("%-9s %8s %8s %8s %9s %9s" %
              ("roi", "ADE med", "ADE mean", "PnP pts", "matches", "fail rate"))
        ref = None
        for r in rois:
            have = [n for n in group if n in results[r]]
            if not have:
                print("%-9s no results" % r); continue
            a = np.array([results[r][n]["ade"] for n in have])
            p = np.array([results[r][n]["pnp"] for n in have])
            m = np.array([results[r][n]["matches"] for n in have])
            fl = np.array([results[r][n]["fail"] for n in have])
            print("%-9s %8.3f %8.3f %8.0f %9.0f %9.3f"
                  % (r, np.median(a), a.mean(), p.mean(), m.mean(), fl.mean()))
            if r == "top":
                ref = (have, a)
        if ref is not None:
            have, a_top = ref
            for r in rois:
                if r == "top":
                    continue
                pair = [n for n in have if n in results[r]]
                if len(pair) < 5:
                    continue
                a0 = np.array([results["top"][n]["ade"] for n in pair])
                a1 = np.array([results[r][n]["ade"] for n in pair])
                d = a1 - a0
                print("  %-7s vs top: median %+7.3f m | better on %3.0f%% of clips "
                      "| n=%d" % (r, np.median(d), 100 * (d < 0).mean(), len(pair)))


if __name__ == "__main__":
    main()
