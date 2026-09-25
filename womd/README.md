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
| `render.py` | one MetaDrive rollout per clip, two 33-frame clips per scenario |
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

## Terrain fixes

Two changes in `render.py` keep the ground under the ego drawn; labels are unaffected.

- **Grass under the ego.** MetaDrive paints roads only inside a 512 m square around the
  ego's first log position, so a fast ego leaves it. The terrain is centred on the ego
  at each clip's middle frame instead (one rollout per clip).
- **Flat grey ground.** The terrain has up to 50 m of relief, and the level ego can end
  up with its camera below the surface, which is not drawn from underneath.
  `TERRAIN_HEIGHT = 0.1` flattens it, which also removes background hills.

## Render environment

`/opt` is not persistent, so the environment has to be rebuilt after a node restart:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3
/opt/miniforge3/bin/conda create -y -n render -c conda-forge python=3.10
R=/opt/miniforge3/envs/render/bin
$R/pip install metadrive-simulator==0.4.3 "protobuf<4"
$R/pip uninstall -y opencv-python && $R/pip install opencv-python-headless
$R/pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/metadriverse/scenarionet.git /opt/scenarionet
$R/pip install --no-deps -e /opt/scenarionet
```

The container has the NVIDIA compute libraries but not EGL. Headless rendering needs the
GL libraries of the host driver (525.147.05) and glvnd:

```bash
apt-get install -y libegl1 libgl1 libglvnd0
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.147.05/NVIDIA-Linux-x86_64-525.147.05.run
sh NVIDIA-Linux-x86_64-525.147.05.run --extract-only
D=NVIDIA-Linux-x86_64-525.147.05; L=/usr/lib/x86_64-linux-gnu
for f in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-tls libGLX_nvidia; do
  cp $D/$f.so.525.147.05 $L/; done
ln -sf libEGL_nvidia.so.525.147.05 $L/libEGL_nvidia.so.0
ln -sf libGLX_nvidia.so.525.147.05 $L/libGLX_nvidia.so.0
mkdir -p /usr/share/glvnd/egl_vendor.d && cp $D/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
ldconfig
```

The render stage holds about 3.5 GB of memory per worker. The container's memory limit
is 240 GiB and includes `/dev/shm`, so `node_build_womd.sh` uses 48 workers; lower it if
`/dev/shm` holds a training cache.
