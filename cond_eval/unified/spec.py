"""One clip format for nuScenes and Waymo.

Both datasets arrive as one uncompressed tar of native front-camera jpgs per
scene, and leave as one tar of clips at the stage-2 training geometry with the
labels a conditioning experiment needs beside them.

    448 x 256, 69 frames, dt = 0.1 s

448 and 256 are both multiples of 32, which Wan2.2's VAE and patch embedding
require, and 69 = 1 + 4*17 gives the causal VAE an integer latent length of 18.
The aspect is 7/4, which neither source matches, so each is centre-cropped to
it first: nuScenes 1600x900 loses 25 px of width, Waymo 1920x1280 loses 183 px
of height.

Three clips per scene, placed at 0, (N-69)/2 and N-69. They tile the scene
exactly -- every frame is used -- with a small overlap at each junction.

**Timing.** Waymo runs at a steady 10 Hz and its frames are taken as they come.
nuScenes averages 12 Hz with gaps alternating 100/100/50 ms, so it is snapped
to a strict 10 Hz grid: the nearest real frame to each target instant, forced
strictly increasing so no frame is emitted twice, with the residual recorded.
Ground truth is always read at the frame actually chosen, so the resampling
adds no error to a displacement metric.

**Distortion is recorded, not removed.** Waymo ships radial and tangential
coefficients; nuScenes' CAM_FRONT calibration carries none. Undistorting would
resample the pixels a second time, so the coefficients travel in the labels and
the images stay as shot.
"""
from __future__ import annotations

import numpy as np

WIDTH, HEIGHT, N_FRAMES, DT = 448, 256, 69, 0.1
CLIPS_PER_SCENE = 3
MOVING_METRES = 10.0            # below this a clip is parked; see DG_CALIBRATION
GRID_US = 100_000               # the 10 Hz grid nuScenes is snapped to


def resize_plan(src_w: int, src_h: int, out_w: int = WIDTH, out_h: int = HEIGHT):
    """Centre crop to the output aspect, then one resize. Returns the crop box
    and the per-axis scale that maps source pixels onto output pixels."""
    want, have = out_w / out_h, src_w / src_h
    if abs(want - have) < 1e-9:
        box = (0, 0, src_w, src_h)
    elif have > want:                       # too wide: crop width
        w = int(round(src_h * want))
        box = ((src_w - w) // 2, 0, w, src_h)
    else:                                   # too tall: crop height
        h = int(round(src_w / want))
        box = (0, (src_h - h) // 2, src_w, h)
    return box, out_w / box[2], out_h / box[3]


def scale_intrinsics(K, crop, sx: float, sy: float) -> np.ndarray:
    """Push a native-resolution calibration through the same crop and resize."""
    x0, y0 = crop[0], crop[1]
    K = np.asarray(K, float).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def uniform_grid(t_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the frames nearest a strict 10 Hz grid spanning the scene,
    plus each pick's distance from its target in milliseconds.

    Two grid points can fall on the same frame where the source stream stalls.
    Emitting it twice would put a hold into footage that real 10 fps never
    contains, so the picks are forced strictly increasing and the extra timing
    error shows up in the residual.
    """
    t = np.asarray(t_us, np.int64)
    n = int((t[-1] - t[0]) // GRID_US) + 1
    targets = t[0] + np.arange(n) * GRID_US
    idx = np.abs(t[None, :] - targets[:, None]).argmin(axis=1)
    for i in range(1, n):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    keep = idx < len(t)
    idx, targets = idx[keep], targets[keep]
    return idx, np.abs(t[idx] - targets) / 1000.0


def clip_starts(n_grid: int, n_frames: int = N_FRAMES,
                k: int = CLIPS_PER_SCENE) -> list[int]:
    """Start indices of the k clips that tile a scene of n_grid frames."""
    if n_grid < n_frames:
        return []
    if k == 1:
        return [(n_grid - n_frames) // 2]
    last = n_grid - n_frames
    return sorted({int(round(i * last / (k - 1))) for i in range(k)})


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def ego_labels(xy: np.ndarray, yaw: np.ndarray, t_us: np.ndarray) -> dict:
    """Everything derived from the pose track of one clip.

    `steps` is the per-adjacent-pair motion expressed in the earlier frame's
    body frame -- forward, lateral, heading change -- which is what an ego-motion
    probe regresses and what composes back into an SE(2) path. Speeds use the
    true timestamps rather than the nominal dt.
    """
    xy = np.asarray(xy, float)
    yaw = np.asarray(yaw, float)
    dt = np.diff(np.asarray(t_us, np.int64)) / 1e6
    d = np.diff(xy, axis=0)
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    fwd = c * d[:, 0] + s * d[:, 1]
    lat = -s * d[:, 0] + c * d[:, 1]
    dth = wrap(np.diff(yaw))
    step = np.linalg.norm(d, axis=1)
    speed = step / np.maximum(dt, 1e-6)
    dist = float(step.sum())
    return {
        "steps": np.stack([fwd, lat, dth], 1).round(6).tolist(),
        "speed_mps": speed.round(4).tolist(),
        "distance_m": round(dist, 3),
        "mean_speed_mps": round(float(speed.mean()), 4),
        "max_speed_mps": round(float(speed.max()), 4),
        "net_yaw_deg": round(float(np.degrees(wrap(yaw[-1] - yaw[0]))), 3),
        "abs_yaw_deg": round(float(np.degrees(np.abs(dth).sum())), 3),
        "is_moving": bool(dist >= MOVING_METRES),
    }


def clip_id(dataset: str, source: str, k: int) -> str:
    return "%s_%s_c%d" % ("nusc" if dataset == "nuscenes" else "wod", source, k)


def shard_name(dataset: str, source: str) -> str:
    return "%s_%s.tar" % ("nusc" if dataset == "nuscenes" else "wod", source)
