"""Cut one scene into clips at the stage-2 geometry and write them as a tar.

Peak local footprint is one source shard plus one output tar, so the whole
build streams: fetch, cut, push, delete.
"""
from __future__ import annotations

import io
import json
import os
import tarfile

import numpy as np
from PIL import Image

from . import spec
from .sources import SceneSource


def plan_clips(s: SceneSource):
    """Frame indices, timing residuals and clip windows for one scene."""
    if s.resample:
        grid, resid = spec.uniform_grid(s.t_us)
    else:
        grid = np.arange(len(s.t_us))
        resid = np.zeros(len(s.t_us))
    starts = spec.clip_starts(len(grid))
    return grid, resid, starts


def _local_frame(xy, yaw):
    """Positions in the clip's first frame body frame, x forward, y left."""
    xy = np.asarray(xy, float) - xy[0]
    c, s = np.cos(-yaw[0]), np.sin(-yaw[0])
    R = np.array([[c, -s], [s, c]])
    return (R @ xy.T).T


def clip_meta(s: SceneSource, grid, resid, start, k, crop, scale, K) -> dict:
    sel = grid[start:start + spec.N_FRAMES]
    t = s.t_us[sel]
    lab = spec.ego_labels(s.xy[sel], s.yaw[sel], t)
    m = {
        "clip_id": spec.clip_id(s.dataset, s.source, k),
        "dataset": s.dataset, "source": s.source,
        "split": s.split, "role": s.role, "clip_index": k,
        "width": spec.WIDTH, "height": spec.HEIGHT,
        "n_frames": spec.N_FRAMES, "dt": spec.DT, "fps": 10.0,
        "src_width": s.src_wh[0], "src_height": s.src_wh[1],
        "crop": list(crop), "scale": [round(scale[0], 8), round(scale[1], 8)],
        "K": np.asarray(K, float).round(6).tolist(),
        "K_native": np.asarray(s.K, float).round(6).tolist(),
        "distortion": [round(v, 9) for v in s.distortion],
        "vehicle_from_camera": s.vehicle_from_camera,
        "resampled_to_10hz": bool(s.resample),
        "frame_files": [s.files[i] for i in sel],
        "t_us": t.tolist(),
        "dt_ms": (np.diff(t) / 1000.0).round(3).tolist(),
        "grid_residual_ms": np.asarray(resid[start:start + spec.N_FRAMES]).round(3).tolist(),
        "ego_xy": s.xy[sel].round(4).tolist(),
        "ego_z": s.z[sel].round(4).tolist(),
        "ego_yaw": s.yaw[sel].round(6).tolist(),
        "cam_xy": s.cam_xy[sel].round(4).tolist(),
        "ego_local_xy": _local_frame(s.xy[sel], s.yaw[sel]).round(4).tolist(),
        "context": s.context,
    }
    m.update(lab)
    return m


def build_scene(s: SceneSource, shard_path: str, out_tar: str,
                jpeg_quality: int = 95) -> list:
    """Write one tar holding every clip of this scene. Returns index records."""
    grid, resid, starts = plan_clips(s)
    if not starts:
        raise ValueError("%s: %d frames on the grid, need %d"
                         % (s.source, len(grid), spec.N_FRAMES))

    # basename -> [(clip, position)], since clips overlap at their junctions
    wanted: dict = {}
    for k, st in enumerate(starts):
        for pos, i in enumerate(grid[st:st + spec.N_FRAMES]):
            wanted.setdefault(s.files[i], []).append((k, pos))

    crop = scale = K = None
    seen = 0
    os.makedirs(os.path.dirname(out_tar), exist_ok=True)
    with tarfile.open(shard_path) as src, tarfile.open(out_tar, "w") as dst:
        for m in src:
            if not m.isfile():
                continue
            slots = wanted.get(os.path.basename(m.name))
            if not slots:
                continue
            img = Image.open(io.BytesIO(src.extractfile(m).read())).convert("RGB")
            if crop is None:
                if (img.width, img.height) != tuple(s.src_wh):
                    s.src_wh = (img.width, img.height)
                crop, sx, sy = spec.resize_plan(img.width, img.height)
                scale = (sx, sy)
                K = spec.scale_intrinsics(s.K, crop, sx, sy)
            x0, y0, w, h = crop
            small = img.crop((x0, y0, x0 + w, y0 + h)).resize(
                (spec.WIDTH, spec.HEIGHT), Image.BICUBIC)
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=jpeg_quality)
            blob = buf.getvalue()
            for k, pos in slots:
                ti = tarfile.TarInfo("clip%d/%05d.jpg" % (k, pos))
                ti.size = len(blob)
                ti.mtime = 0
                dst.addfile(ti, io.BytesIO(blob))
            seen += 1

        if seen != len(wanted):
            raise RuntimeError("%s: %d of %d frames found in shard"
                               % (s.source, seen, len(wanted)))

        records = []
        for k, st in enumerate(starts):
            meta = clip_meta(s, grid, resid, st, k, crop, scale, K)
            blob = json.dumps(meta).encode()
            ti = tarfile.TarInfo("clip%d/meta.json" % k)
            ti.size = len(blob); ti.mtime = 0
            dst.addfile(ti, io.BytesIO(blob))
            r = np.asarray(meta["grid_residual_ms"], float)
            records.append({
                "clip_id": meta["clip_id"], "dataset": s.dataset,
                "source": s.source, "split": s.split, "role": s.role,
                "clip_index": k, "shard": os.path.basename(out_tar),
                "member": "clip%d" % k,
                "n_frames": spec.N_FRAMES, "dt": spec.DT,
                "t0_us": meta["t_us"][0],
                "distance_m": meta["distance_m"],
                "mean_speed_mps": meta["mean_speed_mps"],
                "max_speed_mps": meta["max_speed_mps"],
                "net_yaw_deg": meta["net_yaw_deg"],
                "abs_yaw_deg": meta["abs_yaw_deg"],
                "is_moving": meta["is_moving"],
                "grid_residual_p95_ms": round(float(np.percentile(r, 95)), 3),
                "context": s.context,
            })

        src_blob = json.dumps({
            "dataset": s.dataset, "source": s.source, "split": s.split,
            "role": s.role, "clips": len(starts), "clip_starts": starts,
            "grid_frames": int(len(grid)), "native_frames": int(len(s.t_us)),
            "src_width": s.src_wh[0], "src_height": s.src_wh[1],
            "crop": list(crop), "K": np.asarray(K, float).round(6).tolist(),
            "context": s.context,
        }).encode()
        ti = tarfile.TarInfo("source.json")
        ti.size = len(src_blob); ti.mtime = 0
        dst.addfile(ti, io.BytesIO(src_blob))

    return records
