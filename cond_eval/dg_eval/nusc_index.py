"""Compact per-scene index over the nuScenes tables.

`sample_data.json` is 1.3 GB and `ego_pose.json` 616 MB, so both are parsed once
and reduced to what the trajectory calibration actually reads: for every
CAM_FRONT frame of a scene, its timestamp, its global ego pose, and the camera
calibration it was taken with.

Ego pose is the IMU origin in the global frame, which is *not* where the camera
sits. `camera_xy` composes the sensor extrinsic onto it, so the two can be
compared; the offset is about 1.7 m forward and rotates with heading, which is a
real difference over a turn even though DrivingGen's own pipeline ignores it.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

CAM = "CAM_FRONT"
_TS = re.compile(r"(\d+)(?=\.jpg$)", re.IGNORECASE)


def frame_timestamp(filename: str) -> int:
    """Microsecond timestamp that ends a nuScenes CAM_FRONT filename."""
    m = _TS.search(os.path.basename(filename))
    if m is None:
        raise ValueError(f"no timestamp in {filename!r}")
    return int(m.group(1))


def quat_to_rot(q) -> np.ndarray:
    """nuScenes stores rotation as (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def yaw_from_rot(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class SceneFrames:
    """Every CAM_FRONT frame of one scene, in temporal order."""

    scene: str
    files: list        # basename inside the shard, ordered by timestamp
    t_us: np.ndarray   # (N,) microseconds
    ego_xy: np.ndarray   # (N, 2) global metres, IMU origin
    ego_yaw: np.ndarray  # (N,) radians
    cam_xy: np.ndarray   # (N, 2) global metres, camera optical centre
    K: np.ndarray        # (3, 3) intrinsics at native 1600x900
    cs_token: str

    def __len__(self):
        return len(self.files)


def build_index(raw_dir: str, scenes: list[str] | None = None) -> dict[str, SceneFrames]:
    """Parse the tables and reduce them to one `SceneFrames` per scene.

    Frames are matched to scenes through sample_data -> sample -> scene rather
    than by filename, so sweeps and keyframes are both picked up and nothing
    depends on the shard's contents.
    """
    def load(name):
        with open(os.path.join(raw_dir, name)) as f:
            return json.load(f)

    sensors = {s["token"]: s for s in load("sensor.json")}
    cam_sensor = next(t for t, s in sensors.items() if s["channel"] == CAM)

    calib = {}
    for c in load("calibrated_sensor.json"):
        if c["sensor_token"] != cam_sensor:
            continue
        calib[c["token"]] = {
            "K": np.asarray(c["camera_intrinsic"], float),
            "t": np.asarray(c["translation"], float),
            "R": quat_to_rot(c["rotation"]),
        }

    scene_name = {s["token"]: s["name"] for s in load("scene.json")}
    if scenes is not None:
        keep = set(scenes)
        scene_name = {t: n for t, n in scene_name.items() if n in keep}
    sample_scene = {s["token"]: s["scene_token"] for s in load("sample.json")}

    poses = {}
    for p in load("ego_pose.json"):
        poses[p["token"]] = p

    per_scene: dict[str, list] = {}
    for sd in load("sample_data.json"):
        if sd["calibrated_sensor_token"] not in calib:
            continue
        sc = sample_scene.get(sd["sample_token"])
        name = scene_name.get(sc)
        if name is None:
            continue
        per_scene.setdefault(name, []).append(sd)

    out = {}
    for name, rows in per_scene.items():
        rows.sort(key=lambda r: r["timestamp"])
        cs = calib[rows[0]["calibrated_sensor_token"]]
        files, t_us, exy, eyaw, cxy = [], [], [], [], []
        for r in rows:
            p = poses[r["ego_pose_token"]]
            R = quat_to_rot(p["rotation"])
            T = np.asarray(p["translation"], float)
            files.append(os.path.basename(r["filename"]))
            t_us.append(r["timestamp"])
            exy.append(T[:2])
            eyaw.append(yaw_from_rot(R))
            cxy.append((R @ cs["t"] + T)[:2])
        out[name] = SceneFrames(
            scene=name,
            files=files,
            t_us=np.asarray(t_us, np.int64),
            ego_xy=np.asarray(exy, float),
            ego_yaw=np.asarray(eyaw, float),
            cam_xy=np.asarray(cxy, float),
            K=cs["K"],
            cs_token=rows[0]["calibrated_sensor_token"],
        )
    return out


def save_index(index: dict[str, SceneFrames], path: str) -> None:
    blob = {}
    for name, s in index.items():
        blob[name] = {
            "files": s.files,
            "t_us": s.t_us.tolist(),
            "ego_xy": s.ego_xy.tolist(),
            "ego_yaw": s.ego_yaw.tolist(),
            "cam_xy": s.cam_xy.tolist(),
            "K": s.K.tolist(),
            "cs_token": s.cs_token,
        }
    with open(path, "w") as f:
        json.dump(blob, f)


def load_index(path: str) -> dict[str, SceneFrames]:
    with open(path) as f:
        blob = json.load(f)
    return {
        name: SceneFrames(
            scene=name,
            files=d["files"],
            t_us=np.asarray(d["t_us"], np.int64),
            ego_xy=np.asarray(d["ego_xy"], float),
            ego_yaw=np.asarray(d["ego_yaw"], float),
            cam_xy=np.asarray(d["cam_xy"], float),
            K=np.asarray(d["K"], float),
            cs_token=d["cs_token"],
        )
        for name, d in blob.items()
    }
