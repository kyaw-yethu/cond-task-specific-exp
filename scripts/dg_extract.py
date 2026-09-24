"""Run DrivingGen's ego-trajectory extractor over the built clips.

One process per GPU; the clip list is split by rank so eight ranks cover the set
in one pass. Each clip writes its own npz, so a rank that dies loses only what
it was holding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cond_eval.dg_eval.extract import Extractor, save_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--out", default="/root/driving-gen/traj/dg")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--unidepth", default="/root/driving-gen/ckpt/unidepth-v2-vitl14")
    ap.add_argument("--yolo", default="/root/driving-gen/ckpt/yolov10x.pt")
    # Defaults are the repaired settings, not the released ones. Their published
    # configuration is `--roi top --intrinsics predicted`, which on real footage
    # scores 3.07 m median ADE against 1.13 m for these; see DG_CALIBRATION.md.
    ap.add_argument("--intrinsics", default="calibrated_both",
                    choices=["predicted", "calibrated_slam", "calibrated_both"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roi", default="full", choices=["top", "bottom", "full"],
                    help="which part of the frame may produce features; 'top' is "
                         "what their drive_roi_mask does and costs 12%% of clips")
    ap.add_argument("--names-file", default="",
                    help="restrict to the clips named in this file, one per line")
    args = ap.parse_args()

    if args.names_file:
        names = [l.strip() for l in open(args.names_file) if l.strip()]
    else:
        names = sorted(d for d in os.listdir(args.clips)
                       if os.path.isdir(os.path.join(args.clips, d)))
    if args.limit:
        names = names[:args.limit]
    mine = names[args.rank::args.world]

    out_dir = os.path.join(args.out, args.intrinsics)
    os.makedirs(out_dir, exist_ok=True)

    print("rank %d/%d: %d of %d clips -> %s"
          % (args.rank, args.world, len(mine), len(names), out_dir), flush=True)

    ex = Extractor(args.repo, args.unidepth, args.yolo, roi=args.roi)
    print("rank %d: models loaded" % args.rank, flush=True)

    done, failed, t0 = 0, [], time.time()
    for name in mine:
        dst = os.path.join(out_dir, name + ".npz")
        if os.path.exists(dst):
            done += 1
            continue
        try:
            with open(os.path.join(args.clips, name, "meta.json")) as f:
                meta = json.load(f)
            res = ex.run_clip(
                os.path.join(args.clips, name, "images"),
                np.asarray(meta["K"], float),
                seed=args.seed + abs(hash(name)) % 10000,
                intrinsics_mode=args.intrinsics,
            )
            save_result(res, dst)
            done += 1
            print("rank %d [%3d/%3d] %-12s fail %.3f  depth %5.1fs  slam %5.1fs"
                  % (args.rank, done, len(mine), name, res["fail_rate"],
                     res["sec_depth"], res["sec_slam"]), flush=True)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print("rank %d: %d done, %d failed, %.0f s total"
          % (args.rank, done, len(failed), time.time() - t0), flush=True)
    if failed:
        with open(os.path.join(out_dir, "_failed_rank%d.json" % args.rank), "w") as f:
            json.dump(failed, f, indent=1)


if __name__ == "__main__":
    main()
