"""Build the nuScenes index and cut the evaluation clips.

Runs on the node. The index step parses the two large tables once and caches the
reduction; the clip step reads one shard per scene and writes frames plus the
ground-truth ego and camera tracks beside them.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cond_eval.dg_eval import clips as C
from cond_eval.dg_eval import nusc_index as NI


def _one(job):
    """One clip in a worker. Returns (scene, ok, residuals-or-reason)."""
    shard, scene, geometry, out_dir = job
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        return scene.scene, True, []
    try:
        m = C.build_clip(shard, scene, geometry, out_dir)
        return scene.scene, True, m.grid_residual_ms
    except Exception as e:
        return scene.scene, False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/driving-gen/nuscenes/raw/v1.0-trainval")
    ap.add_argument("--shards", default="/root/driving-gen/nuscenes/shards")
    ap.add_argument("--scenes", default="configs/probe_pool_scenes.txt")
    ap.add_argument("--out", default="/root/driving-gen/clips")
    ap.add_argument("--index", default="/root/driving-gen/nuscenes/index.json")
    ap.add_argument("--geometry", default="dg", choices=sorted(C.GEOMETRIES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    scenes = [l.strip() for l in open(args.scenes) if l.strip()]
    if args.limit:
        scenes = scenes[:args.limit]

    if os.path.exists(args.index):
        print("index: reusing", args.index)
        index = NI.load_index(args.index)
    else:
        t = time.time()
        print("index: parsing tables for %d scenes ..." % len(scenes), flush=True)
        index = NI.build_index(args.raw, scenes)
        NI.save_index(index, args.index)
        print("index: %d scenes in %.0fs -> %s" % (len(index), time.time() - t, args.index))

    n = np.array([len(s) for s in index.values()])
    span = np.array([(s.t_us[-1] - s.t_us[0]) / 1e6 for s in index.values()])
    print("frames/scene: min %d median %d max %d | span %.1f-%.1f s"
          % (n.min(), int(np.median(n)), n.max(), span.min(), span.max()))

    out_root = os.path.join(args.out, args.geometry)
    os.makedirs(out_root, exist_ok=True)
    w, h, nf, hz = C.GEOMETRIES[args.geometry]
    print("geometry %s: %dx%d, %d frames, %s"
          % (args.geometry, w, h, nf, ("%.0f Hz grid" % hz) if hz else "native rate"))

    # Decoding 101 frames of 1600x900 JPEG and resizing them is the whole cost,
    # about 15 s per clip on one core, so the set is cut across a pool.
    jobs = []
    skipped = []
    for name in sorted(index):
        shard = os.path.join(args.shards, "cam_front_%s.tar" % name)
        if not os.path.exists(shard):
            skipped.append((name, "no shard")); continue
        jobs.append((shard, index[name], args.geometry,
                     os.path.join(out_root, name)))

    built, resid = 0, []
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for name, ok, payload in pool.imap_unordered(_one, jobs, chunksize=1):
            if ok:
                built += 1
                resid.extend(payload)
                if built % 25 == 0:
                    print("  %3d/%d  %5.0fs" % (built, len(jobs), time.time() - t0),
                          flush=True)
            else:
                skipped.append((name, payload))

    print("built %d clips, skipped %d, in %.0fs" % (built, len(skipped), time.time() - t0))
    if resid:
        r = np.asarray(resid)
        print("10 Hz grid residual: mean %.1f ms, p95 %.1f ms, max %.1f ms"
              % (r.mean(), np.percentile(r, 95), r.max()))
    for name, why in skipped[:15]:
        print("  skip %s: %s" % (name, why))

    with open(os.path.join(out_root, "_build.json"), "w") as f:
        json.dump({"built": built, "skipped": skipped,
                   "geometry": args.geometry}, f, indent=1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
