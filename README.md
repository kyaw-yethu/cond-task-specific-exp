# cond-drivejepa

Does a driving-specific representation, used as the conditioning signal of a video
diffusion model, give better driving video, in visual quality and in trajectory quality?

## Layout

Four code folders, each split into `lib/` (importable, e.g. `from dg_eval.lib import score`)
and `scripts/` (entry points and node drivers), with its own README.

| folder | |
|---|---|
| [`conditioning/`](conditioning/) | conditioned MiniDiT training and evaluation, stage 1 and 2 |
| [`dg_eval/`](dg_eval/) | DrivingGen trajectory evaluation and its noise floor on nuScenes |
| [`unified/`](unified/) | nuScenes + Waymo clip set at the stage-2 geometry |
| [`womd/`](womd/) | `womd_7hz_f33`, WOMD scenarios rendered in MetaDrive |

`third_party/DrivingGen` is DrivingGen's `drivinggen/` and `scripts/` at 48ed356,
unmodified. `third_party/Task_specific_JDM` is a submodule providing `f_toy`.

## Datasets

**`womd_7hz_f33`**: Waymo Open Motion v1.3.0 `training_20s` scenarios, resampled to
7 Hz and rendered in MetaDrive as ego-camera clips of 33 frames at $96\times96$, with
ego pose, 4 s future waypoints, 32 agent boxes and scene labels per frame. Two clips per
scenario; splits have no scenario in common.

| split | clips | scenarios |
|---|---|---|
| train | 40,000 | 20,000 |
| val | 4,000 | 2,000 |
| test | 4,000 | 2,000 |

About 48 GB uncompressed (`videos.npy` + `annotations.npz` per split). Copies:

- [Google Drive](https://drive.google.com/drive/folders/1kv8P2oRp-c6zdsQd_aaVNAz1QQPCfkAx?usp=sharing),
  `womd_7hz_f33/`: the full folder, including `traj_targets.npz` for Traj-VAE v2, as
  `womd_7hz_f33.tar.zst.part-00..05` (24.7 GB) with `MD5SUMS` and `README.txt`.
- `volume://vessl-storage/yethu-drive/womd/womd_7hz_f33/`: the three splits as
  `{train,val,test}.tar.zst.part-*` (26 GB) with `manifest.json` (md5s), `README.md` and
  `checks.json`; without `traj_targets.npz`, which
  `conditioning/scripts/prep_traj_targets.py` rebuilds.

```bash
cat womd_7hz_f33.tar.zst.part-* | zstd -d | tar -x -C data/     # Drive copy
```

Built by [`womd/`](womd/).

**Unified nuScenes + Waymo clips**: nuScenes CAM_FRONT and Waymo Open Dataset v2 FRONT in
one format, $448\times256$ over 69 frames at 10 Hz (6.9 s), centre-cropped to 7:4, three
clips per scene. Each clip carries intrinsics at that resolution, ego pose, per-step
body-frame increments and caption context. Distortion is recorded, not removed.

| source | scenes | clips | size | split |
|---|---|---|---|---|
| nuScenes | 850 | 2,550 | 7.3 GB | 1,800 train, 750 held-out probe pool |
| Waymo v2 | 1,016 | 3,048 | 8.6 GB | 2,394 training, 606 validation, 48 testing |

Stored at `volume://vessl-storage/yethu-drive/clips/` as one tar per scene
(`{nuscenes,waymo}/shards/`, `clip0..2/{00000..00068}.jpg` plus `meta.json`), a
per-dataset `index.json`, `manifest.json` and `README.md`. Format in
[`unified/docs/FORMAT.md`](unified/docs/FORMAT.md); built by [`unified/`](unified/).

## Checkpoints

Three trained models, in
[Google Drive](https://drive.google.com/drive/folders/1kv8P2oRp-c6zdsQd_aaVNAz1QQPCfkAx?usp=sharing).
Full settings, logs and md5s are in [`conditioning/runs/`](conditioning/runs/).

| | MiniWan | Probe | Traj-VAE v2 |
|---|---|---|---|
| file | `miniwan/best.pt` | `probe/probe.pt` | `traj_vae/best.pt` |
| params | 5.60M | 1.42M | 2.39M encoder + 0.26M heads |
| input | 33 frames, $96\times96$, 7 Hz (`womd_7hz_f33`) | 16 frames, $96\times96$ (`driving_wp_f16`) | 33 frames, $96\times96$, 7 Hz (`womd_7hz_f33`) |
| output | latent $16\times9\times12\times12$ | 15 ego-pose increments | $r$, latent $16\times9\times12\times12$ |

**MiniWan**: a small Wan2.1-style causal 3D video VAE with width 20 and 16 latent
channels. It compresses 8x in space and 4x in time, with causal convolutions, so latent
frame $t$ only sees raw frames up to its own chunk. It is the video latent space for
`womd_7hz_f33`, and its encoder initialises Traj-VAE v2. It was trained from scratch
for 50 epochs on 4 GPUs (effective batch 24) with MSE, LPIPS $\times0.05$ and KL
$\times3\times10^{-4}$. Validation, 4,000 clips: PSNR 31.33 dB, LPIPS 0.0198.

**Probe**: a CNN that reads the ego trajectory out of a clip. Each adjacent frame pair is
stacked into 6 channels and passed through a shared 2D trunk ($96\to3$, width 256). Two
temporal 1D convolutions follow, then a head that predicts
$(\Delta_\text{forward}, \Delta_\text{lateral}, \Delta_\text{heading})$ per step. The
increments are composed as SE(2) transforms into a trajectory. It is the stage-1
trajectory extractor, used to score the ego path in real or generated clips. It shares no
weights with any conditioning representation. It was trained from scratch for 20 epochs
on one GPU with half the clips passed through a VAE round trip, so it tolerates generated
frames. Its floor on the test split: ADE 0.225 m on real frames, 0.274 m after the VAE
round trip.

**Traj-VAE v2**: MiniWan's encoder, cloned and fine-tuned, with training-only heads on
its latent $r$ (the encoder's $\mu$). The heads predict per-frame ego motion, 4 s
waypoints from the first $k\in\{1,2,3\}$ latent frames (6 hypotheses, loss on the
closest), a $12\times12$ agent occupancy grid, and scene labels (lane curvature,
intersection, traffic light). The heads are dropped after training. $r$ is the V
conditioning source: a driving-specific representation that stays informative from one
context latent frame. It was trained for 30 epochs on 4 GPUs with encoder lr
$5\times10^{-5}$ and head lr $10^{-3}$. Best epoch is 19: waypoint ADE 3.61 / 2.79 /
2.83 m at $k$ = 1 / 2 / 3.

## Setup

```bash
git clone --recursive <this repo> && cd cond-drivejepa
pip install -r requirements.txt               # after torch, see the header of the file
bash dg_eval/scripts/setup_third_party.sh     # UniDepth, yolov10, MTR from DrivingGen
bash dg_eval/scripts/node_dg_env.sh           # system libs + no-deps installs for the extractor
bash conditioning/scripts/setup_cudafix.sh    # cluster-3090 only, fixes CUDA error 804
```

On cluster-3090 (driver 525) torch must stay at `2.3.1+cu121` with `numpy<2` and no
xformers. Run every script from the repository root: relative defaults resolve against
it, and `data/`, `checkpoints/`, `cache/` and `out/` there are gitignored.

```bash
pytest
```

`third_party/DrivingGen` is Apache-2.0; `conditioning/lib/traj_metrics.py` ports from it.
