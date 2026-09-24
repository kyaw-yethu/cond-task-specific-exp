"""Build womd_7hz_f33 (DATASET_PLAN.md). Stages, each resumable:

  convert   raw shards -> resampled ScenarioNet pickles + window stats   (render env)
  select    window pairs for train/val, slice-quota test set             (any env)
  render    MetaDrive rollouts -> videos.npy + annotations.npz per split (render env)
  finalize  pixel_stats, checks, compressed copy to yethu-drive          (training env)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAW = "/opt/womd/raw"
SD_ROOT = "/opt/womd/sd"
WORK = "/opt/womd/work"
OUT = "data/womd_7hz_f33"
SHARDS = dict(train=list(range(0, 420)), val=list(range(900, 950)), test=list(range(800, 900)) + list(range(950, 1000)))


def _convert(shard):
    from womd.lib.convert import convert_shard
    p = os.path.join(RAW, "training_20s.tfrecord-%05d-of-01000" % shard)
    t = time.time()
    rows = convert_shard(p, SD_ROOT, shard)
    return shard, len(rows), time.time() - t


def stage_convert(workers):
    shards = [s for r in SHARDS.values() for s in r]
    t0 = time.time()
    n = 0
    with Pool(workers, maxtasksperchild=20) as pool:
        for i, (shard, k, dt) in enumerate(pool.imap_unordered(_convert, shards)):
            n += k
            if (i + 1) % 20 == 0 or i + 1 == len(shards):
                print("convert %d/%d shards, %d scenarios, %.0fs" % (i + 1, len(shards), n, time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["convert", "select", "render", "finalize"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    if a.stage == "convert":
        stage_convert(a.workers)
    elif a.stage == "select":
        from womd.lib.select import stage_select
        stage_select(SD_ROOT, WORK, SHARDS)
    elif a.stage == "render":
        from womd.lib.render import stage_render
        stage_render(SD_ROOT, WORK, OUT, a.split, a.workers)
    elif a.stage == "finalize":
        from womd.lib.finalize import stage_finalize
        stage_finalize(WORK, OUT, a.split)
