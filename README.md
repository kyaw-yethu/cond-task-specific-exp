# cond-drivejepa

Does a driving-specific representation, used as the conditioning signal of a video
diffusion model, give better driving video, in visual quality and in trajectory quality?

Each condition trains the same MiniDiT from scratch with a different representation fed
through cross-attention, then every run is scored on two axes: video quality (FVD,
PSNR, SSIM, LPIPS) and the ego trajectory recovered from the generated frames, scored
with DrivingGen's trajectory metrics (ADE/FDE, comfort, curvature, FTD).

| tag | conditioning |
|---|---|
| N | none |
| D | Drive-JEPA, domain-adapted encoder |
| T | Drive-JEPA, task encoder |
| V | Traj-VAE latent |
| G | V-JEPA2 ViT-H |

## Layout

```
cond_eval/            library code
  cond_dit.py           ConditionedDiT, cross-attention conditioner
  r_sources.py          conditioning sources (encoders per tag)
  cached.py             fixed-k latent + token cache dataset
  dit_eval.py           video metrics + trajectory metrics for one run
  traj_metrics.py       DrivingGen trajectory metrics, ported and tested
  pose_probe.py         ego-motion probe used as the stage-1 trajectory extractor
  traj_v2.py            Traj-VAE v2
  dg_eval/              DrivingGen's extractor and scorer driven over nuScenes
  unified/              nuScenes + Waymo unified clip set (448x256, 69 frames)
  womd/                 WOMD -> womd_7hz_f33 rendering
scripts/              entry points; node_*.sh are the drivers used on the workspace
  data/                 Waymo Open v2 fetchers
configs/              DiT configs, probe scene pool, ROI subset
docs/                 plans, results and the DrivingGen calibration write-up
tests/                pytest for traj_metrics, pose_probe, dit_eval
third_party/
  DrivingGen/           drivinggen/ and scripts/ from upstream at 48ed356, unmodified
  Task_specific_JDM/    submodule, provides f_toy (MiniDiT, VAE, datasets)
```

Read `docs/STAGE1_PLAN.md`, `docs/PROBE.md` and `docs/STAGE2_PLAN.md` for the design,
`docs/DG_CALIBRATION.md` for the trajectory-metric noise floor.

## Setup

```bash
git clone --recursive <this repo> && cd cond-drivejepa
pip install -r requirements.txt       # after torch, see the header of the file
bash scripts/setup_third_party.sh     # UniDepth, yolov10, MTR from DrivingGen
bash scripts/node_dg_env.sh           # system libs + no-deps installs for the extractor
bash scripts/setup_cudafix.sh         # cluster-3090 only, fixes CUDA error 804
```

On the cluster-3090 nodes (driver 525), torch must stay at `2.3.1+cu121` with
`numpy<2` and no xformers; check `torch.cuda.is_available()` after any pip install.
The I3D weights for FVD download on first use.

Run everything from the repository root. Relative defaults (`third_party/...`,
`data/`, `checkpoints/`, `cache/`, `out/`) resolve against it; `data/`,
`checkpoints/`, `cache/` and `out/` are gitignored and are expected to exist locally.
DrivingGen evaluation data (nuScenes, clips, UniDepth and YOLO weights) defaults to
`/root/driving-gen/`, overridable per script with `--clips`, `--raw`, `--unidepth`,
`--yolo`.

## Pipeline

Stage 1, the conditioning screen, on Waymo `driving_wp_f16`:

```bash
bash scripts/node_train_probe.sh                          # ego-motion probe + floors
bash scripts/node_build_and_train.sh 0 N none             # cache, then train one tag
bash scripts/node_build_and_train.sh 1 T drivejepa drivejepa_T_vitl256
GPU=5 bash scripts/node_eval_cond.sh T                    # both axes, 4,000 test clips
python scripts/make_results.py --out-root out --proj .    # RESULTS.md
```

`scripts/node_orchestrate.sh` chains evaluation, comparison figures and results for
all tags once training finishes.

DrivingGen trajectory evaluation on nuScenes:

```bash
python scripts/dg_build.py                 # index + clips at DG (1024x576, 101 f) and S2 (448x256, 69 f)
bash scripts/node_dg_run_all.sh            # extractor, all geometries x intrinsics modes
python scripts/dg_score.py                 # ADE/FDE and quality against ground truth
python scripts/dg_ftd.py                   # FTD and its floor
python scripts/dg_prepare_generated.py     # generated videos -> clips the suite reads
```

Unified nuScenes + Waymo clip set: `bash scripts/node_build_unified.sh`, format in
`docs/UNIFIED_CLIPS.md`.

## Tests

```bash
pytest
```

## Licenses

`third_party/DrivingGen` is Apache-2.0 (see its `LICENSE`); `cond_eval/traj_metrics.py`
ports functions from it.
