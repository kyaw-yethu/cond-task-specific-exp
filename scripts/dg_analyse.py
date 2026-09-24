"""Per-clip distribution behind the summary means.

The means are tail-dominated, so the mode comparison is unreadable without the
median and the size of the tail beside it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/dg_calibration")
    ap.add_argument("--tag", default="dg")
    ap.add_argument("--ref", default="ego")
    args = ap.parse_args()

    rows = []
    for mode in ("predicted", "calibrated_slam", "calibrated_both"):
        f = os.path.join(args.dir, "per_clip_%s_%s_%s_fixed_scale.npz"
                         % (args.tag, mode, args.ref))
        g = os.path.join(args.dir, "per_clip_%s_%s_%s_scaled.npz"
                         % (args.tag, mode, args.ref))
        if not os.path.exists(f):
            continue
        d, ds = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
        ade, fr = d["ade"], d["fail_rate"]
        rows.append((mode, ade, ds["ade"], fr, d["names"]))

    print("%-18s %7s %7s %7s %7s %7s %7s" %
          ("mode", "mean", "median", "p25", "p75", "p95", ">10m"))
    for mode, ade, _, _, _ in rows:
        print("%-18s %7.3f %7.3f %7.3f %7.3f %7.3f %6d%%" %
              (mode, ade.mean(), np.median(ade), np.percentile(ade, 25),
               np.percentile(ade, 75), np.percentile(ade, 95),
               round(100 * (ade > 10).mean())))

    print("\nscale-free ADE (per-clip optimal scale)")
    print("%-18s %7s %7s %7s" % ("mode", "mean", "median", "p95"))
    for mode, _, sade, _, _ in rows:
        print("%-18s %7.3f %7.3f %7.3f" %
              (mode, sade.mean(), np.median(sade), np.percentile(sade, 95)))

    if rows:
        base_names = rows[0][4]
        print("\npaired deltas against predicted, median ADE per clip")
        base = rows[0][1]
        for mode, ade, _, _, names in rows[1:]:
            if list(names) != list(base_names):
                print("  %-18s clip sets differ, skipping" % mode); continue
            d = ade - base
            better = (d < 0).mean()
            print("  %-18s median %+6.3f m | better on %.0f%% of clips"
                  % (mode, np.median(d), 100 * better))

        print("\nworst clips under predicted")
        order = np.argsort(base)[::-1][:6]
        for i in order:
            print("  %-14s ADE %8.2f m  fail_rate %.3f" %
                  (base_names[i], base[i], rows[0][3][i]))

        print("\nclips whose fixed-scale ADE exceeds the clip's own path length")
        # a trajectory that collapses scores an ADE near the distance travelled
        print("  (proxy: ADE > 10 m) predicted %d / %d"
              % (int((base > 10).sum()), base.size))


if __name__ == "__main__":
    main()
