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
