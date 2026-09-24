# unified

nuScenes CAM_FRONT and Waymo v2 FRONT in one clip format at the stage-2 geometry,
448x256 over 69 frames at 10 Hz, stored at
`volume://vessl-storage/yethu-drive/clips/` as one tar per scene with per-clip
`meta.json` and a per-dataset `index.json`. Format in `docs/FORMAT.md`.

## lib

| module | |
|---|---|
| `spec.py` | the geometry and the label maths |
| `sources.py` | each dataset's tables reduced to one per-scene record |
| `build.py` | one scene cut into clips and written as a tar |
| `volume.py` | scoped read and write access to the VESSL volume |

## scripts

| script | |
|---|---|
| `wod_fetch_front.py` | stream Waymo v2 `camera_image` from GCS, keep FRONT, push per-scene tars |
| `wod_fetch_labels.py` | Waymo v2 non-pixel components, one tar per component |
| `build_unified.py` | build the clip set for one dataset and push it |
| `finalize_unified.py` | check the set on the volume, write its README and manifest |
| `node_build_unified.sh` | both datasets, detached, logs in `/root/unified` |

```bash
bash unified/scripts/node_build_unified.sh
python unified/scripts/finalize_unified.py
```

The fetchers take no arguments and push to the volume as soon as they run.
