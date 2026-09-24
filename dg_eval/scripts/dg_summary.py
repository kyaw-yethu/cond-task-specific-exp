"""The calibration's headline table, and the paired tests behind it.

`dg_analyse` reports over every clip and `dg_full_table` over the moving ones;
this puts the two together in the shape the report carries, and adds the paired
comparisons between configurations, which are what say whether a difference is
real rather than a shift of the mean by a handful of broken clips.

A clip counts as moving when its true path is at least 10 m. Roughly a quarter
of nuScenes is effectively parked and scores near zero whatever the estimator
does, so an all-clip figure partly measures how many stationary scenes the set
happens to contain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

LABEL = {
    ("%s", "predicted"): "top + predicted (as published)",
    ("%s", "calibrated_slam"): "top + calibrated_slam",
    ("%s", "calibrated_both"): "top + calibrated_both",
    ("full_%s", "predicted"): "full + predicted",
    ("full_%s", "calibrated_both"): "full + calibrated_both (repaired)",
}


def cells(geom):
    return [(t % geom, m, lab) for (t, m), lab in LABEL.items()]


def load(base, tag, mode, ref="ego"):
    f = os.path.join(base, "per_clip_%s_%s_%s_fixed_scale.npz" % (tag, mode, ref))
    g = os.path.join(base, "per_clip_%s_%s_%s_scaled.npz" % (tag, mode, ref))
    if not os.path.exists(f):
        return None
    d, ds = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
    return {"names": [str(x) for x in d["names"]], "ade": d["ade"],
            "sade": ds["ade"], "sr": d["success_rate"], "fail": d["fail_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/dg_calibration")
    ap.add_argument("--clips", default="/root/driving-gen/clips/stage2")
    ap.add_argument("--geom", default="stage2")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    rows = [(t, m, lab, load(args.base, t, m)) for t, m, lab in cells(args.geom)]
    rows = [r for r in rows if r[3]]
    names = rows[0][3]["names"]

    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            p = np.asarray(json.load(f)["ego_xy"], float)
        driven.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    driven = np.asarray(driven)
    mv = driven >= 10.0
    print("%d clips, %d moving, %d parked | path length median %.1f m"
          % (len(names), mv.sum(), (~mv).sum(), np.median(driven[mv])))

    scales = {}
    for t, m, lab, _ in rows:
        rep = json.load(open(os.path.join(args.base, "report_%s.json" % t)))
        scales[(t, m)] = rep["modes"][m]["cuts"]["ego/scaled"]["_scale"]["median"]

    print("\n=== moving clips ===")
    hdr = ("configuration", "ADEmed", "ADEmean", "p95", ">10m", "sADE", "SR@3m",
           "scale", "fail")
    print("%-36s%8s%8s%8s%7s%8s%8s%8s%7s" % hdr)
    for t, m, lab, d in rows:
        a, sa, sr, fr = d["ade"][mv], d["sade"][mv], d["sr"][mv], d["fail"][mv]
        print("%-36s%8.3f%8.3f%8.2f%6.0f%%%8.3f%8.3f%8.3f%6.1f%%"
              % (lab, np.median(a), a.mean(), np.percentile(a, 95),
                 100 * (a > 10).mean(), np.median(sa), sr.mean(),
                 scales[(t, m)], 100 * fr.mean()))

    print("\n=== all clips ===")
    print("%-36s%8s%8s%8s" % ("configuration", "ADEmed", "ADEmean", "SR@3m"))
    for t, m, lab, d in rows:
        print("%-36s%8.3f%8.3f%8.3f"
              % (lab, np.median(d["ade"]), d["ade"].mean(), d["sr"].mean()))

    def paired(a_key, b_key, what):
        A = next(d for t, m, l, d in rows if (t, m) == a_key)
        B = next(d for t, m, l, d in rows if (t, m) == b_key)
        d = B["ade"][mv] - A["ade"][mv]
        print("  %-34s median %+7.3f m | better on %3.0f%% of clips"
              % (what, np.median(d), 100 * (d < 0).mean()))

    g, fg = args.geom, "full_" + args.geom
    print("\n=== paired, against the published configuration ===")
    paired((g, "predicted"), (fg, "predicted"), "whole frame alone")
    paired((g, "predicted"), (g, "calibrated_both"), "real intrinsics alone")
    paired((g, "predicted"), (fg, "calibrated_both"), "both")
    paired((g, "predicted"), (g, "calibrated_slam"), "real intrinsics in the SLAM only")

    if args.report:
        out = {"geom": args.geom, "n_clips": len(names),
               "n_moving": int(mv.sum()),
               "median_path_m": float(np.median(driven[mv])), "cells": {}}
        for t, m, lab, d in rows:
            a, sa = d["ade"][mv], d["sade"][mv]
            out["cells"][lab] = {
                "ade_median": float(np.median(a)), "ade_mean": float(a.mean()),
                "ade_p95": float(np.percentile(a, 95)),
                "frac_over_10m": float((a > 10).mean()),
                "sade_median": float(np.median(sa)),
                "sr3m": float(d["sr"][mv].mean()),
                "scale_median": scales[(t, m)],
                "fail_rate": float(d["fail"][mv].mean()),
                "ade_median_all": float(np.median(d["ade"])),
                "ade_mean_all": float(d["ade"].mean())}
        with open(args.report, "w") as f:
            json.dump(out, f, indent=1)
        print("\nwrote", args.report)


if __name__ == "__main__":
    main()
