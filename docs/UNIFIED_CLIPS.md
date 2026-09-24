# Unified driving clips, 448x256 over 69 frames

nuScenes CAM_FRONT and Waymo Open Dataset v2 FRONT, cut to one format and one
geometry so a model can be trained or scored on either without a second data
path.

    448 x 256 pixels, 69 frames, dt = 0.1 s, 6.9 s per clip

448 and 256 are both multiples of 32, which Wan2.2's VAE and patch embedding
require, and 69 = 1 + 4*17 gives the causal VAE an integer latent length of 18.
The aspect is 7/4, which neither source matches, so each is centre-cropped to it
first: nuScenes 1600x900 loses 25 px of width, Waymo 1920x1280 loses 183 px of
height. Intrinsics are pushed through the same crop and resize and travel with
each clip.

Three clips per scene, at frames 0, (N-69)/2 and N-69. They tile the scene, so
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
| `K` | 3x3 intrinsics at 448x256. `K_native` is the same before the crop |
| `distortion` | k1, k2, p1, p2, k3. All zero for nuScenes |
| `vehicle_from_camera` | the sensor extrinsic, R and t, as the dataset stores it |
| `frame_files` | the native filenames the 69 frames came from |
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

| | scenes | clips | moving | train / eval / test | size |
|---|---|---|---|---|---|
| **nuscenes** | 850 | 2550 | 2005 | 1800 / 750 / 0 | 7.3 GB |
| **waymo** | 1016 | 3048 | 2296 | 2394 / 606 / 48 | 8.6 GB |
| **total** | 1866 | 5598 | 4301 | | 16.0 GB |

Median clip: 36 m of path at 5.4 m/s for nuScenes, 38 m at 5.6 m/s for Waymo.
The nuScenes 10 Hz resampling leaves a p95 residual of 50 ms; Waymo needs none.

Splits are each dataset's own. nuScenes `probe_pool` is the 250-scene held-out
set the trajectory calibration uses, and the other 600 scenes are `train`.
Waymo's `training`, `validation` and `testing` map to `train`, `eval` and `test`.

Built by `cond_eval/unified/` and `scripts/build_unified.py` in
`cond-drivejepa-exp`.
