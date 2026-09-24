"""Cut calibrated evaluation clips out of the CAM_FRONT shards.

Two geometries, because two different questions are being asked of the same
footage:

`dg`      101 frames on a uniform 10 Hz grid at 1024x576. This is DrivingGen's
          own setting, so a number measured here can be read beside their
          published table. nuScenes is 1600x900, whose aspect ratio matches
          1024x576 exactly, so the frame is a plain resize with no crop.

`stage2`  69 frames on the same uniform 10 Hz grid at 448x256, the training
          geometry stage 2 uses. Both sides must be multiples of 32 for
          Wan2.2's VAE and patch embedding, and 69 = 1 + 4*17 is what the
          causal VAE needs for an integer latent length. 448/256 = 7/4 is not
          1600/900, so the crop comes off the width: 1575x900.

CAM_FRONT runs at 12 Hz on average but is irregularly spaced, with gaps
alternating 100/100/50 ms, so the 10 Hz grid takes the nearest real frame to
each target instant and records the residual. Ground truth is read at the
frames actually chosen, never at the nominal grid time, so the resampling adds
no error to a displacement metric; it only makes the nominal dt approximate for
the derivative-based ones.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

from .nusc_index import SceneFrames

GEOMETRIES = {
    # name:      (out_w, out_h, n_frames, uniform_hz)
    "dg":        (1024, 576, 101, 10.0),
    "stage2":    (448,  256,  69, 10.0),
}


@dataclass
class ClipMeta:
    scene: str
    geometry: str
    width: int
    height: int
    n_frames: int
    frame_idx: list        # indices into the scene's frame list
    t_us: list             # true microsecond timestamps of those frames
    grid_residual_ms: list # |chosen - target| on the 10 Hz grid, 0 when native
    K: list                # 3x3 intrinsics for the emitted resolution
    crop: list             # [x0, y0, w, h] taken from the 1600x900 source
    ego_xy: list
    cam_xy: list
    ego_yaw: list


def _resize_plan(src_w: int, src_h: int, out_w: int, out_h: int):
    """Centre crop to the output aspect, then a single resize. Returns the crop
    box and the scale that maps source pixels onto output pixels."""
    want = out_w / out_h
    have = src_w / src_h
    if abs(want - have) < 1e-6:
        box = (0, 0, src_w, src_h)
    elif have > want:                      # source too wide: crop width
        w = int(round(src_h * want))
        box = ((src_w - w) // 2, 0, w, src_h)
    else:                                  # source too tall: crop height
        h = int(round(src_w / want))
        box = (0, (src_h - h) // 2, src_w, h)
    return box, out_w / box[2], out_h / box[3]


def scale_intrinsics(K: np.ndarray, crop, sx: float, sy: float) -> np.ndarray:
    """Push a 1600x900 calibration through the same crop and resize."""
    x0, y0, _, _ = crop
    K = np.asarray(K, float).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def pick_frames(scene: SceneFrames, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices for one clip, centred in the scene, plus the 10 Hz
    residual in milliseconds."""
    _, _, n, hz = GEOMETRIES[geometry]
    t = scene.t_us
    if hz is None:
        if len(t) < n:
            raise ValueError(f"{scene.scene}: {len(t)} frames < {n}")
        start = (len(t) - n) // 2
        idx = np.arange(start, start + n)
        return idx, np.zeros(n)

    step_us = int(round(1e6 / hz))
    span = (n - 1) * step_us
    if t[-1] - t[0] < span:
        raise ValueError(f"{scene.scene}: spans {(t[-1]-t[0])/1e6:.1f}s < {span/1e6:.1f}s")
    t0 = t[0] + ((t[-1] - t[0]) - span) // 2
    targets = t0 + np.arange(n) * step_us
    idx = np.abs(t[None, :] - targets[:, None]).argmin(axis=1)

    # The 12 Hz stream is irregular, with gaps alternating 100/100/50 ms, so
    # two grid points can land on the same frame. Emitting it twice would put a
    # stall into the clip that real 10 fps footage never contains and would
    # inflate the floor being measured, so the picks are forced strictly
    # increasing instead and the extra timing error is reported.
    for i in range(1, n):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    if idx[-1] >= len(t):
        raise ValueError(f"{scene.scene}: ran out of frames forcing a strict 10 Hz grid")
    return idx, np.abs(t[idx] - targets) / 1000.0


def build_clip(shard_path: str, scene: SceneFrames, geometry: str,
               out_dir: str, jpeg_quality: int = 95) -> ClipMeta:
    out_w, out_h, n, _ = GEOMETRIES[geometry]
    idx, resid = pick_frames(scene, geometry)

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    wanted = {scene.files[i]: k for k, i in enumerate(idx)}
    crop = scale = None
    written = 0
    with tarfile.open(shard_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            k = wanted.get(base)
            if k is None:
                continue
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            if crop is None:
                crop, sx, sy = _resize_plan(img.width, img.height, out_w, out_h)
                scale = (sx, sy)
            x0, y0, w, h = crop
            img = img.crop((x0, y0, x0 + w, y0 + h)).resize((out_w, out_h), Image.BICUBIC)
            img.save(os.path.join(img_dir, "%05d.jpg" % k), quality=jpeg_quality)
            written += 1
    if written != n:
        raise RuntimeError(f"{scene.scene}: wrote {written}/{n} frames")

    meta = ClipMeta(
        scene=scene.scene,
        geometry=geometry,
        width=out_w,
        height=out_h,
        n_frames=n,
        frame_idx=idx.tolist(),
        t_us=scene.t_us[idx].tolist(),
        grid_residual_ms=resid.tolist(),
        K=scale_intrinsics(scene.K, crop, *scale).tolist(),
        crop=list(crop),
        ego_xy=scene.ego_xy[idx].tolist(),
        cam_xy=scene.cam_xy[idx].tolist(),
        ego_yaw=scene.ego_yaw[idx].tolist(),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(asdict(meta), f)
    return meta
