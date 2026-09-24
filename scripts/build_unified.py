"""Build the unified clip set and push it to the volume.

    volume://vessl-storage/yethu-drive/clips/
        README.md
        nuscenes/index.json     waymo/index.json
        nuscenes/shards/*.tar   waymo/shards/*.tar

One tar per source scene, holding that scene's three clips at 448x256 over 69
frames plus their labels. Nothing large is ever held on disk: a worker fetches
one source shard into /dev/shm, cuts it, uploads the result and deletes both.

Resumable. A scene whose tar is already on the volume and whose records are
already cached locally is skipped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cond_eval.unified import build as B
from cond_eval.unified import sources as S
from cond_eval.unified import spec
from cond_eval.unified import volume as V

DEST = "clips"
SRC_PREFIX = {"nuscenes": "assets/nuscenes/cam_front_shards",
              "waymo": "assets/waymo/cam_front_shards"}
WORK = "/dev/shm/unified"
CACHE = "/root/unified"

_G = {}


def _init(dataset: str, keep_local: str):
    # federation is an API call per handle, so the pool does not ask all at once
    time.sleep(random.random() * 10)
    _G["vol_r"] = V.open_volume(False)
    _G["vol_w"] = V.open_volume(True)
    _G["dataset"] = dataset
    _G["keep_local"] = keep_local
    with open(os.path.join(CACHE, "sources_%s.pkl" % dataset), "rb") as f:
        _G["src"] = pickle.load(f)
    _G["avail"] = V.listing(_G["vol_r"], SRC_PREFIX[dataset])


def _reopen():
    _G["vol_r"] = V.open_volume(False)
    _G["vol_w"] = V.open_volume(True)


def _one(name: str):
    """Fetch, cut, push, clean. Returns (name, ok, payload)."""
    t0 = time.time()
    ds = _G["dataset"]
    s = _G["src"][name]
    rec_path = os.path.join(CACHE, "records", ds, name + ".json")
    if os.path.exists(rec_path):
        return name, True, ("cached", 0.0, 0)

    shard_local = os.path.join(_G["keep_local"] or (WORK + "/src"), s.shard)
    out_tar = os.path.join(WORK, "out", ds, spec.shard_name(ds, name))
    fetched = False
    try:
        size = _G["avail"].get(s.shard)
        if size is None:
            return name, False, "no shard on volume"
        if not (os.path.exists(shard_local) and
                os.path.getsize(shard_local) == size):
            try:
                V.fetch(_G["vol_r"], SRC_PREFIX[ds], s.shard, size, shard_local)
            except Exception:
                _reopen()
                V.fetch(_G["vol_r"], SRC_PREFIX[ds], s.shard, size, shard_local)
            fetched = True

        records = B.build_scene(s, shard_local, out_tar)
        dest = "%s/%s/shards/%s" % (DEST, ds, os.path.basename(out_tar))
        try:
            nbytes = V.push(_G["vol_w"], out_tar, dest)
        except Exception:
            _reopen()
            nbytes = V.push(_G["vol_w"], out_tar, dest)

        os.makedirs(os.path.dirname(rec_path), exist_ok=True)
        with open(rec_path, "w") as f:
            json.dump(records, f)
        return name, True, ("built", time.time() - t0, nbytes)
    except Exception as e:
        traceback.print_exc()
        return name, False, "%s: %s" % (type(e).__name__, e)
    finally:
        if os.path.exists(out_tar):
            os.remove(out_tar)
        if fetched and not _G["keep_local"] and os.path.exists(shard_local):
            os.remove(shard_local)


def load_sources(dataset: str, args) -> dict:
    path = os.path.join(CACHE, "sources_%s.pkl" % dataset)
    if os.path.exists(path) and not args.rebuild_index:
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    if dataset == "nuscenes":
        ev = {l.strip() for l in open(args.eval_scenes) if l.strip()}
        src = S.nuscenes_sources(args.nusc_raw, ev)
    else:
        src = S.waymo_sources(args.wod_meta)
    print("  parsed %d %s scenes in %.0fs" % (len(src), dataset, time.time() - t),
          flush=True)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(src, f)
    return src


def report(dataset: str, src: dict):
    n = np.array([len(s.t_us) for s in src.values()])
    gaps = []
    for s in list(src.values())[:50]:
        gaps.append(np.diff(s.t_us) / 1000.0)
    g = np.concatenate(gaps)
    print("  frames/scene: min %d median %d max %d" % (n.min(), np.median(n), n.max()))
    print("  native gap ms: median %.1f  p05 %.1f  p95 %.1f"
          % (np.median(g), np.percentile(g, 5), np.percentile(g, 95)))
    roles = {}
    for s in src.values():
        roles[s.role] = roles.get(s.role, 0) + 1
    print("  roles:", roles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nuscenes", "waymo"])
    ap.add_argument("--nusc-raw", default="/root/driving-gen/nuscenes/raw/v1.0-trainval")
    ap.add_argument("--wod-meta", default="/dev/shm/wod_meta")
    ap.add_argument("--eval-scenes",
                    default="configs/probe_pool_scenes.txt")
    ap.add_argument("--keep-local", default="",
                    help="directory of already-downloaded source shards to reuse")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="reparse the source tables instead of reusing the cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(WORK, "src"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "out", args.dataset), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "records", args.dataset), exist_ok=True)

    print("### sources: %s" % args.dataset, flush=True)
    src = load_sources(args.dataset, args)
    report(args.dataset, src)

    names = sorted(src)
    if args.limit:
        names = names[:args.limit]
    todo = [n for n in names
            if not os.path.exists(os.path.join(CACHE, "records", args.dataset,
                                               n + ".json"))]
    print("### %d scenes, %d to build, %d workers"
          % (len(names), len(todo), args.workers), flush=True)

    ok, failed, nbytes, t0 = 0, [], 0, time.time()
    if todo:
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.dataset, args.keep_local)) as pool:
            for name, good, payload in pool.imap_unordered(_one, todo, chunksize=1):
                if good:
                    ok += 1
                    if isinstance(payload, tuple):
                        nbytes += payload[2]
                    if ok % 25 == 0:
                        el = time.time() - t0
                        print("  %4d/%d  %5.0fs  %5.2f GB up  ETA %.2f h"
                              % (ok, len(todo), el, nbytes / 1e9,
                                 (len(todo) - ok) * el / max(ok, 1) / 3600),
                              flush=True)
                else:
                    failed.append((name, payload))
                    print("  FAIL %s: %s" % (name, payload), flush=True)

    print("built %d, failed %d, %.2f GB uploaded, %.0fs"
          % (ok, len(failed), nbytes / 1e9, time.time() - t0), flush=True)

    # merge the per-scene records into one index and push it
    records = []
    rdir = os.path.join(CACHE, "records", args.dataset)
    for f in sorted(os.listdir(rdir)):
        with open(os.path.join(rdir, f)) as fh:
            records.extend(json.load(fh))
    index = {
        "dataset": args.dataset,
        "geometry": {"width": spec.WIDTH, "height": spec.HEIGHT,
                     "n_frames": spec.N_FRAMES, "dt": spec.DT, "fps": 10.0,
                     "clips_per_scene": spec.CLIPS_PER_SCENE},
        "n_scenes": len(os.listdir(rdir)), "n_clips": len(records),
        "moving_clips": int(sum(r["is_moving"] for r in records)),
        "shard_prefix": "%s/%s/shards" % (DEST, args.dataset),
        "failed": failed,
        "clips": records,
    }
    out = os.path.join(CACHE, "index_%s.json" % args.dataset)
    with open(out, "w") as f:
        json.dump(index, f)
    vol = V.open_volume(True)
    V.push(vol, out, "%s/%s/index.json" % (DEST, args.dataset))
    print("index: %d clips (%d moving) -> %s/%s/index.json"
          % (len(records), index["moving_clips"], DEST, args.dataset), flush=True)


if __name__ == "__main__":
    main()
