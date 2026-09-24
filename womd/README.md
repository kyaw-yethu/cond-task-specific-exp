# womd

`womd_7hz_f33`: Waymo Open Motion training_20s scenarios resampled to 7 Hz and
rendered in MetaDrive into 33-frame clips with ego labels. Used to train Traj-VAE v2
and the video VAE in `conditioning`. The packed copy lives at
`volume://vessl-storage/yethu-drive/womd/womd_7hz_f33/`.

## lib

| module | |
|---|---|
| `tfrecord.py` | minimal TFRecord reader, no tensorflow |
| `resample.py` | ScenarioNet scenario onto an even time grid |
| `convert.py` | one tfrecord shard into resampled scenarios and per-window stats |
| `stats.py` | per-window statistics for window and test-slice selection |
| `select.py` | scenarios and window pairs per split |
| `render.py` | one MetaDrive rollout per scenario, two 33-frame clips |
| `finalize.py` | labels stacked into the `DiskDataset` layout, with checks |

## scripts

| script | |
|---|---|
| `fetch_womd.py` | fetch the training_20s shards |
| `build_womd.py` | convert, select, render, finalize; each stage resumable |
| `node_build_womd.sh` | render all three splits, then finalize |
| `spotcheck_womd.py` | 20 clips per split, five frames each |
| `pack_womd.py` | zstd-tar each split and push to the volume |

`fetch_womd.py`, `spotcheck_womd.py` and `pack_womd.py` take no arguments and act as
soon as they run; `pack_womd.py` overwrites the volume's `manifest.json`. Rendering
needs the MetaDrive environment at `/opt/miniforge3/envs/render`.
