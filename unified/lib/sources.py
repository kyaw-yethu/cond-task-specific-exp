"""The two datasets reduced to one per-scene record.

Everything that differs between nuScenes and Waymo is resolved here -- table
formats, pose conventions, which camera is the front one, where the split comes
from -- so the builder downstream sees a single shape.

`SceneSource.xy` is the vehicle origin in each dataset's own world frame, in
metres, with `yaw` the heading in that frame. Neither frame is shared between
scenes and nothing downstream needs it to be: every label is either a
difference between consecutive frames or is re-expressed relative to the clip's
first frame.
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
from dataclasses import dataclass, field

import numpy as np

CAM_FRONT_WOD = 1               # Waymo's front camera id
_TS = re.compile(r"(\d+)(?=\.jpg$)", re.IGNORECASE)


def frame_timestamp(name: str) -> int:
    m = _TS.search(os.path.basename(name))
    if m is None:
        raise ValueError("no timestamp in %r" % name)
    return int(m.group(1))


def quat_to_rot(q) -> np.ndarray:
    """nuScenes stores rotation as (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def yaw_from_rot(R) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class SceneSource:
    dataset: str            # nuscenes | waymo
    source: str             # scene-0001 | waymo context name
    split: str              # the dataset's own label
    role: str               # train | eval | test
    shard: str              # tar basename in the source shard directory
    files: list             # member basenames, temporal order
    t_us: np.ndarray
    xy: np.ndarray          # (N, 2) vehicle origin, world metres
    z: np.ndarray           # (N,)
    yaw: np.ndarray         # (N,)
    cam_xy: np.ndarray      # (N, 2) camera optical centre, world metres
    K: np.ndarray           # (3, 3) at native resolution
    distortion: list        # [k1, k2, p1, p2, k3]
    src_wh: tuple           # native (width, height)
    vehicle_from_camera: dict  # {"R": 3x3, "t": 3} as the dataset stores it
    context: dict = field(default_factory=dict)   # weather, time of day, place
    resample: bool = True   # snap to the 10 Hz grid

    def __len__(self):
        return len(self.files)


# --------------------------------------------------------------- nuScenes ---

def nuscenes_sources(raw_dir: str, eval_scenes: set) -> dict:
    """One record per scene, over every scene the tables describe.

    Frames are matched to scenes through sample_data -> sample -> scene rather
    than by filename, so keyframes and sweeps are both picked up.
    """
    def load(name):
        with open(os.path.join(raw_dir, name)) as f:
            return json.load(f)

    sensors = {s["token"]: s for s in load("sensor.json")}
    cam_token = next(t for t, s in sensors.items() if s["channel"] == "CAM_FRONT")

    calib = {}
    for c in load("calibrated_sensor.json"):
        if c["sensor_token"] == cam_token:
            calib[c["token"]] = {"K": np.asarray(c["camera_intrinsic"], float),
                                 "t": np.asarray(c["translation"], float),
                                 "R": quat_to_rot(c["rotation"])}

    logs = {l["token"]: l for l in load("log.json")}
    scenes = {}
    for s in load("scene.json"):
        lg = logs.get(s["log_token"], {})
        scenes[s["token"]] = {
            "name": s["name"],
            "description": s.get("description", ""),
            "location": lg.get("location", ""),
            "vehicle": lg.get("vehicle", ""),
            "date": lg.get("date_captured", ""),
        }
    sample_scene = {s["token"]: s["scene_token"] for s in load("sample.json")}
    poses = {p["token"]: p for p in load("ego_pose.json")}

    per_scene = {}
    for sd in load("sample_data.json"):
        if sd["calibrated_sensor_token"] not in calib:
            continue
        sc = sample_scene.get(sd["sample_token"])
        if sc in scenes:
            per_scene.setdefault(sc, []).append(sd)

    out = {}
    for tok, rows in per_scene.items():
        info = scenes[tok]
        rows.sort(key=lambda r: r["timestamp"])
        cs = calib[rows[0]["calibrated_sensor_token"]]
        files, t, xy, z, yaw, cxy = [], [], [], [], [], []
        for r in rows:
            p = poses[r["ego_pose_token"]]
            R = quat_to_rot(p["rotation"])
            T = np.asarray(p["translation"], float)
            files.append(os.path.basename(r["filename"]))
            t.append(r["timestamp"])
            xy.append(T[:2]); z.append(T[2]); yaw.append(yaw_from_rot(R))
            cxy.append((R @ cs["t"] + T)[:2])
        name = info["name"]
        out[name] = SceneSource(
            dataset="nuscenes", source=name,
            split="probe_pool" if name in eval_scenes else "train",
            role="eval" if name in eval_scenes else "train",
            shard="cam_front_%s.tar" % name,
            files=files, t_us=np.asarray(t, np.int64),
            xy=np.asarray(xy, float), z=np.asarray(z, float),
            yaw=np.asarray(yaw, float), cam_xy=np.asarray(cxy, float),
            K=cs["K"], distortion=[0.0] * 5, src_wh=(1600, 900),
            vehicle_from_camera={"R": cs["R"].tolist(), "t": cs["t"].tolist()},
            context={"description": info["description"],
                     "location": info["location"],
                     "vehicle": info["vehicle"],
                     "date": info["date"]},
            resample=True,
        )
    return out


# ------------------------------------------------------------------ Waymo ---

def _pq(blob):
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(blob))


def waymo_sources(meta_dir: str) -> dict:
    """One record per segment, from the four component tars.

    Waymo runs a steady 10 Hz, so `resample` is off and the frames are taken as
    they come; the builder still records the inter-frame spacing so the
    assumption is checkable.
    """
    import pyarrow.parquet as pq

    def members(tar):
        with tarfile.open(os.path.join(meta_dir, tar)) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    yield m.name, tf.extractfile(m).read()

    calib = {}
    for name, blob in members("camera_calibration.tar"):
        t = _pq(blob).to_pydict()
        ctx = t["key.segment_context_name"][0]
        for i, cam in enumerate(t["key.camera_name"]):
            if cam != CAM_FRONT_WOD:
                continue
            g = lambda k: t["[CameraCalibrationComponent].%s" % k][i]
            K = np.array([[g("intrinsic.f_u"), 0.0, g("intrinsic.c_u")],
                          [0.0, g("intrinsic.f_v"), g("intrinsic.c_v")],
                          [0.0, 0.0, 1.0]])
            ext = np.asarray(g("extrinsic.transform"), float).reshape(4, 4)
            calib[ctx] = {
                "K": K,
                "distortion": [g("intrinsic.k1"), g("intrinsic.k2"),
                               g("intrinsic.p1"), g("intrinsic.p2"),
                               g("intrinsic.k3")],
                "wh": (int(g("width")), int(g("height"))),
                "R": ext[:3, :3], "t": ext[:3, 3],
            }

    ctx_stats = {}
    for name, blob in members("stats.tar"):
        t = _pq(blob).to_pydict()
        ctx = t["key.segment_context_name"][0]
        ctx_stats[ctx] = {
            "weather": t["[StatsComponent].weather"][0],
            "time_of_day": t["[StatsComponent].time_of_day"][0],
            "location": t["[StatsComponent].location"][0],
        }

    out = {}
    for name, blob in members("vehicle_pose.tar"):
        split = name.split("/")[0]
        t = _pq(blob).to_pydict()
        ctx = t["key.segment_context_name"][0]
        if ctx not in calib:
            continue
        ts = np.asarray(t["key.frame_timestamp_micros"], np.int64)
        tf_ = np.asarray(t["[VehiclePoseComponent].world_from_vehicle.transform"],
                         float).reshape(-1, 4, 4)
        order = np.argsort(ts)
        ts, tf_ = ts[order], tf_[order]
        xy = tf_[:, :2, 3]
        z = tf_[:, 2, 3]
        yaw = np.arctan2(tf_[:, 1, 0], tf_[:, 0, 0])
        c = calib[ctx]
        cam = (tf_[:, :3, :3] @ c["t"][None, :, None]).squeeze(-1) + tf_[:, :3, 3]
        out[ctx] = SceneSource(
            dataset="waymo", source=ctx, split=split,
            role={"training": "train", "validation": "eval",
                  "testing": "test"}[split],
            shard="cam_front_%s.tar" % ctx,
            files=["%s__CAM_FRONT__%d.jpg" % (ctx, v) for v in ts],
            t_us=ts, xy=xy, z=z, yaw=yaw, cam_xy=cam[:, :2],
            K=c["K"], distortion=[float(v) for v in c["distortion"]],
            src_wh=c["wh"],
            vehicle_from_camera={"R": c["R"].tolist(), "t": c["t"].tolist()},
            context=ctx_stats.get(ctx, {}),
            resample=False,
        )
    return out
