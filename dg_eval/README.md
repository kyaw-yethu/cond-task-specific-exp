# dg_eval

DrivingGen's trajectory evaluation run on nuScenes CAM_FRONT, with ground-truth ego pose,
to measure the extractor's own error floor before any generator is scored. Two
geometries: DG (1024x576, 101 frames), DrivingGen's own, and S2 (448x256, 69 frames),
the stage-2 training geometry. Write-up in `docs/DG_CALIBRATION.md`.

Uses the vendored `third_party/DrivingGen/drivinggen`; UniDepth, yolov10 and MTR come
from `scripts/setup_third_party.sh`, and `scripts/node_dg_env.sh` installs them.

## lib

| module | |
|---|---|
| `nusc_index.py` | compact per-scene index over the nuScenes tables |
| `clips.py` | calibrated clips cut from the CAM_FRONT shards |
| `extract.py` | DrivingGen's ego-trajectory extractor, with failure flag, intrinsics mode and seed |
| `score.py` | DrivingGen's alignment and metric functions |

## scripts

**Pipeline**: `dg_build.py`, `dg_extract.py`, `dg_score.py`, `dg_ftd.py`,
`dg_prepare_generated.py` (generated videos into clips the suite reads).

**Analysis**: `dg_analyse.py`, `dg_full_table.py`, `dg_summary.py`, `dg_diagnose.py`,
`dg_roi_test.py`.

**Figures**: `plot_dg_traj.py`, `plot_dg_overlays.py`, `dg_viz.py`, `dg_viz3d.py`.

**Setup and drivers**: `setup_third_party.sh`, `node_dg_env.sh`, `node_dg_run_all.sh`
(every geometry against every intrinsics mode).

```bash
bash dg_eval/scripts/setup_third_party.sh && bash dg_eval/scripts/node_dg_env.sh
python dg_eval/scripts/dg_build.py
bash dg_eval/scripts/node_dg_run_all.sh
python dg_eval/scripts/dg_score.py
python dg_eval/scripts/dg_ftd.py
```

Data defaults to `/root/driving-gen/`. `configs/` holds the held-out probe scene pool
and the ROI subset.
