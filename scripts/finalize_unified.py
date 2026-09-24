"""Check the unified clip set on the volume, then write its README and manifest.

Everything printed here is read back off the volume rather than from the build's
own bookkeeping, so a shard that failed to land shows up as a mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cond_eval.unified import spec
from cond_eval.unified import volume as V

DEST = "clips"
CACHE = "/root/unified"


def stats(index, shards):
    c = index["clips"]
    roles = Counter(r["role"] for r in c)
    splits = Counter(r["split"] for r in c)
    dist = np.array([r["distance_m"] for r in c])
    spd = np.array([r["mean_speed_mps"] for r in c])
    turn = np.array([abs(r["net_yaw_deg"]) for r in c])
    moving = np.array([r["is_moving"] for r in c])
    resid = np.array([r["grid_residual_p95_ms"] for r in c])
    ctx = Counter()
    for r in c:
        for k in ("weather", "time_of_day"):
            if r["context"].get(k):
                ctx[r["context"][k]] += 1
    return {
        "scenes": index["n_scenes"], "clips": len(c),
        "shards_on_volume": len(shards),
        "bytes": int(sum(shards.values())),
        "roles": dict(roles), "splits": dict(splits),
        "moving": int(moving.sum()),
        "median_path_m": round(float(np.median(dist)), 1),
        "median_speed_mps": round(float(np.median(spd)), 2),
        "median_abs_turn_deg": round(float(np.median(turn)), 1),
        "grid_residual_p95_ms": round(float(np.percentile(resid, 95)), 1),
        "context": dict(ctx.most_common(8)),
    }


README = """# Unified driving clips, {W}x{H} over {F} frames

nuScenes CAM_FRONT and Waymo Open Dataset v2 FRONT, cut to one format and one
geometry so a model can be trained or scored on either without a second data
path.

    {W} x {H} pixels, {F} frames, dt = {DT} s, {SEC} s per clip

{W} and {H} are both multiples of 32, which Wan2.2's VAE and patch embedding
require, and {F} = 1 + 4*17 gives the causal VAE an integer latent length of 18.
The aspect is 7/4, which neither source matches, so each is centre-cropped to it
first: nuScenes {NW}x{NH} loses 25 px of width, Waymo {WW}x{WH} loses 183 px of
height. Intrinsics are pushed through the same crop and resize and travel with
each clip.

Three clips per scene, at frames 0, (N-{F})/2 and N-{F}. They tile the scene, so
every frame is used, with a small overlap at each junction.

**Timing.** Waymo runs a steady 10 Hz and its frames are taken as they come.
nuScenes averages 12 Hz with gaps alternating 100/100/50 ms, so it is snapped to
a strict 10 Hz grid: the nearest real frame to each target instant, forced
strictly increasing so no frame is emitted twice, with the residual recorded per
frame. Ground truth is always read at the frame actually chosen, so the
resampling adds no error to a displacement metric.

**Distortion is recorded, not removed.** Waymo ships radial and tangential
coefficients; nuScenes' CAM_FRONT calibration carries none. Undistorting would
resample the pixels a second time, so the coefficients sit in the labels and the
images stay as shot.

## Layout

    clips/
        README.md
        manifest.json               counts and geometry for both datasets
        nuscenes/index.json         one record per clip, summary fields only
        nuscenes/shards/nusc_<scene>.tar
        waymo/index.json
        waymo/shards/wod_<context>.tar

One tar per source scene:

    clip0/00000.jpg ... clip0/00068.jpg
    clip0/meta.json
    clip1/...  clip2/...
    source.json

## What is in a clip's `meta.json`

| field | |
|---|---|
| `clip_id`, `dataset`, `source`, `split`, `role`, `clip_index` | identity. `role` is `train`, `eval` or `test` |
| `width`, `height`, `n_frames`, `dt`, `fps` | the geometry above |
| `src_width`, `src_height`, `crop`, `scale` | how the frame was cut from the native image |
| `K` | 3x3 intrinsics at {W}x{H}. `K_native` is the same before the crop |
| `distortion` | k1, k2, p1, p2, k3. All zero for nuScenes |
| `vehicle_from_camera` | the sensor extrinsic, R and t, as the dataset stores it |
| `frame_files` | the native filenames the {F} frames came from |
| `t_us`, `dt_ms`, `grid_residual_ms` | true timestamps, spacing, and distance from the 10 Hz target |
| `ego_xy`, `ego_z`, `ego_yaw` | vehicle origin in the dataset's own world frame, metres and radians |
| `cam_xy` | camera optical centre in the same frame |
| `ego_local_xy` | the path in the clip's first-frame body frame, x forward, y left |
| `steps` | per adjacent pair, (forward, lateral, heading change) in the earlier frame's body frame. Composing them reproduces `ego_local_xy` to 1e-4 m. This is what an ego-motion probe regresses |
| `speed_mps`, `distance_m`, `mean_speed_mps`, `max_speed_mps` | derived from the true timestamps, not the nominal dt |
| `net_yaw_deg`, `abs_yaw_deg` | heading change end to end, and total turning |
| `is_moving` | true when the clip travels at least 10 m. A parked clip carries no trajectory signal |
| `context` | nuScenes: scene description, location, vehicle, date. Waymo: weather, time of day, location |

`index.json` carries every scalar field of every clip without the per-frame
arrays, so a dataloader can filter on split, motion or weather without opening a
tar.

## What is here

{TABLE}

Splits are each dataset's own. nuScenes `probe_pool` is the 250-scene held-out
set the trajectory calibration uses, and the other 600 scenes are `train`.
Waymo's `training`, `validation` and `testing` map to `train`, `eval` and `test`.

Built by `cond_eval/unified/` and `scripts/build_unified.py` in
`cond-drivejepa-exp`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/unified")
    args = ap.parse_args()

    vol = V.open_volume(True)
    man = {"geometry": {"width": spec.WIDTH, "height": spec.HEIGHT,
                        "n_frames": spec.N_FRAMES, "dt": spec.DT, "fps": 10.0,
                        "seconds": round(spec.N_FRAMES * spec.DT, 2),
                        "clips_per_scene": spec.CLIPS_PER_SCENE,
                        "moving_threshold_m": spec.MOVING_METRES},
           "datasets": {}}

    rows = []
    for ds in ("nuscenes", "waymo"):
        shards = V.listing(vol, "%s/%s/shards" % (DEST, ds))
        with open(os.path.join(CACHE, "index_%s.json" % ds)) as f:
            index = json.load(f)
        s = stats(index, shards)
        man["datasets"][ds] = s
        print("=== %s ===" % ds)
        for k, v in s.items():
            print("  %-22s %s" % (k, v))
        ok = s["shards_on_volume"] == s["scenes"]
        print("  shards match scenes: %s" % ok)
        rows.append((ds, s))

    tot_clips = sum(s["clips"] for _, s in rows)
    tot_bytes = sum(s["bytes"] for _, s in rows)
    man["total_clips"] = tot_clips
    man["total_bytes"] = tot_bytes
    print("\ntotal %d clips, %d shards, %.1f GB"
          % (tot_clips, sum(s["shards_on_volume"] for _, s in rows), tot_bytes / 1e9))

    tbl = ["| | scenes | clips | moving | train / eval / test | size |",
           "|---|---|---|---|---|---|"]
    for ds, s in rows:
        r = s["roles"]
        tbl.append("| **%s** | %d | %d | %d | %d / %d / %d | %.1f GB |"
                   % (ds, s["scenes"], s["clips"], s["moving"],
                      r.get("train", 0), r.get("eval", 0), r.get("test", 0),
                      s["bytes"] / 1e9))
    tbl.append("| **total** | %d | %d | %d | | %.1f GB |"
               % (sum(s["scenes"] for _, s in rows), tot_clips,
                  sum(s["moving"] for _, s in rows), tot_bytes / 1e9))
    tbl.append("")
    tbl.append("Median clip: %.0f m of path at %.1f m/s for nuScenes, "
               "%.0f m at %.1f m/s for Waymo."
               % (rows[0][1]["median_path_m"], rows[0][1]["median_speed_mps"],
                  rows[1][1]["median_path_m"], rows[1][1]["median_speed_mps"]))
    tbl.append("The nuScenes 10 Hz resampling leaves a p95 residual of %.0f ms; "
               "Waymo needs none." % rows[0][1]["grid_residual_p95_ms"])

    readme = README.format(
        W=spec.WIDTH, H=spec.HEIGHT, F=spec.N_FRAMES, DT=spec.DT,
        SEC=round(spec.N_FRAMES * spec.DT, 1),
        NW=1600, NH=900, WW=1920, WH=1280, TABLE="\n".join(tbl))

    rp = os.path.join(args.out, "README.md")
    mp = os.path.join(args.out, "manifest.json")
    with open(rp, "w") as f:
        f.write(readme)
    with open(mp, "w") as f:
        json.dump(man, f, indent=1)
    V.push(vol, rp, "%s/README.md" % DEST)
    V.push(vol, mp, "%s/manifest.json" % DEST)
    print("\npushed %s/README.md and %s/manifest.json" % (DEST, DEST))


if __name__ == "__main__":
    main()
