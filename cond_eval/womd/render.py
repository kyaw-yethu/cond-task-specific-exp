"""Render selected scenarios: one MetaDrive rollout per scenario, two 33-frame clips out.

Rendering follows f_toy.datasets.waymo.scenario.render_scenario (same camera, 96x96
RGB, lidar off, ego log replay). Differences: the scenario is the 7 Hz resample, one
env.step is 7 x 0.02 s of physics, the rollout runs to the end of the log even when
MetaDrive reports arrive_dest (it does so at step 0 for a parked ego), and the labels
carry the extra DATASET_PLAN.md section 5 fields.

Clip j of scenario i lands at row 2 i + j of the split videos.npy, so workers
write pixels straight into the memmap; labels go to one npz per scenario under
<work>/ann_<split>/ and are stacked by finalize.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "third_party/Task_specific_JDM")

CLIP = 33
HORIZON = 28            # 4 s of future ego pose at 7 Hz
N_AGENTS = 32
AGENT_MAX_M = 60.0
IMG = 96
CAMERA_POS = (0.0, 1.2, 1.3)
CAMERA_HPR = (0.0, -15.0, 0.0)
CAMERA_FOV = 60
TARGET_MAX_DIST = 60.0
TARGET_CONE_DEG = 30.0
INTENT_DEG = 20.0
CONTEXT_RAW = 5         # raw frames 0..4 are context at k=2
GPUS = (0, 1, 2, 3)
CHUNK = 2000

TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST, TYPE_NONE = 0, 1, 2, -1
TL_CODE = {None: -1, "LANE_STATE_UNKNOWN": 0,
           "LANE_STATE_STOP": 1, "LANE_STATE_ARROW_STOP": 1, "LANE_STATE_FLASHING_STOP": 1,
           "LANE_STATE_CAUTION": 2, "LANE_STATE_ARROW_CAUTION": 2, "LANE_STATE_FLASHING_CAUTION": 2,
           "LANE_STATE_GO": 3, "LANE_STATE_ARROW_GO": 3}
INTENT = dict(straight=0, left=1, right=2)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_env(db_dir, n):
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "load-display p3headlessgl")
    loadPrcFileData("", "egl-device-index %d" % GPUS[os.getpid() % len(GPUS)])
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    return ScenarioEnv(dict(
        data_directory=db_dir, num_scenarios=n, start_scenario_index=0,
        agent_policy=ReplayEgoCarPolicy, show_terrain=True, show_sidewalk=True,
        image_observation=True, image_on_cuda=False, use_render=False, decision_repeat=7,
        vehicle_config=dict(image_source="rgb_camera", lidar=dict(num_lasers=0, distance=0),
                            side_detector=dict(num_lasers=0)),
        sensors={"rgb_camera": (RGBCamera, IMG, IMG)},
        show_interface=False, show_logo=False, show_fps=False, window_size=(IMG, IMG),
        horizon=1000, crash_vehicle_done=False, out_of_route_done=False))


# ---------------------------------------------------------------- map context
class LaneIndex:
    """Nearest-lane lookup on the scenario map, in log coordinates (MetaDrive
    keeps Waymo coordinates unchanged; checked to 0.0 m on replay)."""

    def __init__(self, sd):
        pts, hd, lid, kap, seg = [], [], [], [], []
        self.interp = {}
        for k, f in sd["map_features"].items():
            if "LANE" not in str(f.get("type", "")) or "polyline" not in f:
                continue
            p = np.asarray(f["polyline"], dtype=np.float64)[:, :2]
            if len(p) < 5:
                continue
            d = np.diff(p, axis=0)
            h = np.arctan2(d[:, 1], d[:, 0])
            h = np.append(h, h[-1])
            ds = np.append(np.hypot(d[:, 0], d[:, 1]), 1e-3)
            dh = _wrap(np.roll(h, -2) - np.roll(h, 2))
            s = np.convolve(ds, np.ones(4), mode="same")
            kp = np.where(s > 1e-3, dh / np.maximum(s, 1e-3), 0.0)
            kp[:2] = kp[2]
            kp[-3:] = kp[-4]
            pts.append(p); hd.append(h); kap.append(kp); lid.append(np.full(len(p), len(seg)))
            seg.append(str(k))
            self.interp[str(k)] = bool(f.get("interpolating", False))
        self.ok = bool(pts)
        if self.ok:
            self.p = np.concatenate(pts); self.h = np.concatenate(hd)
            self.k = np.concatenate(kap); self.l = np.concatenate(lid)
        self.ids = seg

    def query(self, xy, heading, max_d=3.0, max_dh=np.radians(45)):
        if not self.ok:
            return None, 0.0, False
        d = np.hypot(self.p[:, 0] - xy[0], self.p[:, 1] - xy[1])
        d = np.where(np.abs(_wrap(self.h - heading)) > max_dh, np.inf, d)
        i = int(np.argmin(d))
        if not np.isfinite(d[i]) or d[i] > max_d:
            return None, 0.0, False
        lane = self.ids[self.l[i]]
        return lane, float(self.k[i]), self.interp[lane]


def scene_context(sd, idx):
    """Per log index: ego-lane traffic light code, lane curvature (1/m, left
    positive), intersection flag (Waymo interpolating lane), lane found."""
    ego = sd["tracks"][sd["metadata"]["sdc_id"]]["state"]
    lanes = LaneIndex(sd)
    tl_by_lane = {}
    for dm in sd["dynamic_map_states"].values():
        if dm.get("lane") is not None:
            tl_by_lane[str(dm["lane"])] = dm["state"]["object_state"]
    n = len(idx)
    tl = np.full(n, -1, np.int64); kap = np.zeros(n, np.float32)
    inter = np.zeros(n, np.float32); found = np.zeros(n, np.float32)
    for j, t in enumerate(idx):
        lane, k, it = lanes.query(ego["position"][t, :2], ego["heading"][t])
        if lane is None:
            continue
        found[j], kap[j], inter[j] = 1.0, k, float(it)
        if lane in tl_by_lane:
            tl[j] = TL_CODE.get(tl_by_lane[lane][t], 0)
    return tl, kap, inter, found


# ---------------------------------------------------------------- rollout
def _type_code(obj, classes):
    Ped, Cyc, Veh = classes
    if isinstance(obj, Ped):
        return TYPE_PEDESTRIAN
    if isinstance(obj, Cyc):
        return TYPE_CYCLIST
    if isinstance(obj, Veh):
        return TYPE_VEHICLE
    return TYPE_NONE


def rollout(env, index):
    from f_toy.datasets.metadrive.scenario import _project_bbox_2d
    from metadrive.component.traffic_participants.cyclist import Cyclist
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    classes = (Pedestrian, Cyclist, BaseVehicle)

    env.reset(seed=index)
    cam = env.engine.get_sensor("rgb_camera")
    cam.cam.setPos(*CAMERA_POS)
    cam.cam.setHpr(*CAMERA_HPR)
    lens = cam.cam.node().getLens()
    sd = env.engine.data_manager.current_scenario
    L = int(sd["length"])
    id_map = env.engine.traffic_manager._obj_id_to_scenario_id

    frames, ego_pos, ego_head, ego_speed, coll, steps = [], [], [], [], [], []
    per_frame = []
    for t in range(L - 1):
        o, r, term, trunc, info = env.step([0.0, 0.0])
        frame = (np.clip(o["image"][..., -1][..., ::-1], 0.0, 1.0) * 255).astype(np.uint8)
        frames.append(frame.transpose(2, 0, 1))
        ego = env.agent
        ex, ey = ego.position
        eh = ego.heading_theta
        ego_pos.append((ex, ey)); ego_head.append(eh); ego_speed.append(ego.speed)
        coll.append(float(bool(info.get("crash_vehicle", False))))
        steps.append(env.engine.episode_step)
        c, s = np.cos(eh), np.sin(eh)
        ev = np.array(ego.velocity)
        recs = {}
        for oid, obj in env.engine.get_objects().items():
            if obj is ego or not isinstance(obj, classes):
                continue
            dx, dy = np.array(obj.position) - np.array(ego.position)
            dist = float(np.hypot(dx, dy))
            if dist > AGENT_MAX_M:
                continue
            fwd, lat = dx * c + dy * s, -dx * s + dy * c
            cx, cy, w, hp, ok = _project_bbox_2d(cam.cam, lens, obj, IMG)
            rv = np.array(obj.velocity) - ev
            closing = -(dx * rv[0] + dy * rv[1]) / (dist + 1e-6)
            try:
                tid = int(id_map.get(oid))
            except (TypeError, ValueError):
                tid = -1
            recs[oid] = dict(fwd=fwd, lat=lat, dist=dist, ok=bool(ok and fwd > 0), bbox=(cx, cy, w, hp),
                             dz=obj.origin.getPos()[-1] - ego.origin.getPos()[-1],
                             heading_rel=_wrap(obj.heading_theta - eh),
                             dims=(obj.LENGTH, obj.WIDTH, obj.HEIGHT), speed=obj.speed,
                             type=_type_code(obj, classes), closing=closing, tid=tid)
        per_frame.append(recs)
    return sd, dict(video=np.stack(frames, axis=1), ego_position=np.array(ego_pos, np.float32),
                    ego_heading=np.array(ego_head, np.float32), ego_speed=np.array(ego_speed, np.float32),
                    collision=np.array(coll, np.float32), steps=np.array(steps), per_frame=per_frame)


def clip_labels(sd, R, t0, entry, scen_index, slice_code):
    T = CLIP
    frames = range(t0, t0 + T)
    pf = R["per_frame"]
    cone = np.radians(TARGET_CONE_DEG)
    # stage-1 target: nearest in-cone object at the first frame of the clip
    target, best = None, TARGET_MAX_DIST
    for oid, rec in pf[t0].items():
        if rec["fwd"] <= 0 or abs(np.arctan2(rec["lat"], rec["fwd"])) > cone:
            continue
        if rec["dist"] < best:
            target, best = oid, rec["dist"]
    out = dict(
        target_visible=np.zeros(T, np.float32), target_bbox_2d=np.zeros((T, 4), np.float32),
        target_position_3d=np.zeros((T, 3), np.float32), target_heading_rel=np.zeros(T, np.float32),
        target_dims=np.zeros((T, 3), np.float32), target_speed=np.zeros(T, np.float32),
        target_type=np.full(T, TYPE_NONE, np.int64), lead_distance=np.full(T, -1.0, np.float32),
        ttc=np.full(T, 999.0, np.float32), ttc_valid=np.zeros(T, np.float32))
    if target is not None:
        for i, t in enumerate(frames):
            rec = pf[t].get(target)
            if rec is None:
                continue
            out["target_visible"][i] = float(rec["ok"])
            out["target_bbox_2d"][i] = rec["bbox"]
            out["target_position_3d"][i] = (rec["fwd"], rec["lat"], rec["dz"])
            out["target_heading_rel"][i] = rec["heading_rel"]
            out["target_dims"][i] = rec["dims"]
            out["target_speed"][i] = rec["speed"]
            out["target_type"][i] = rec["type"]
            out["lead_distance"][i] = rec["dist"]
            if rec["closing"] > 0.1:
                out["ttc"][i] = min(rec["dist"] / rec["closing"], 999.0)
                out["ttc_valid"][i] = 1.0
    # all agents: the N_AGENTS nearest within AGENT_MAX_M, per frame
    a_id = np.full((T, N_AGENTS), -1, np.int64); a_type = np.full((T, N_AGENTS), TYPE_NONE, np.int64)
    a_pos = np.zeros((T, N_AGENTS, 3), np.float32); a_head = np.zeros((T, N_AGENTS), np.float32)
    a_speed = np.zeros((T, N_AGENTS), np.float32); a_box = np.zeros((T, N_AGENTS, 4), np.float32)
    a_vis = np.zeros((T, N_AGENTS), np.float32); a_valid = np.zeros((T, N_AGENTS), np.float32)
    a_dims = np.zeros((T, N_AGENTS, 3), np.float32)
    for i, t in enumerate(frames):
        near = sorted(pf[t].values(), key=lambda r: r["dist"])[:N_AGENTS]
        for j, rec in enumerate(near):
            a_id[i, j] = rec["tid"]; a_type[i, j] = rec["type"]
            a_pos[i, j] = (rec["fwd"], rec["lat"], rec["dz"]); a_head[i, j] = rec["heading_rel"]
            a_speed[i, j] = rec["speed"]; a_box[i, j] = rec["bbox"]; a_vis[i, j] = float(rec["ok"])
            a_valid[i, j] = 1.0; a_dims[i, j] = rec["dims"]
    # future ego
    L = R["video"].shape[1]
    fpos = np.zeros((HORIZON, 2), np.float32); fhead = np.zeros(HORIZON, np.float32)
    fval = np.zeros(HORIZON, np.float32)
    fs = t0 + T
    n = max(0, min(HORIZON, L - fs))
    if n:
        fpos[:n] = R["ego_position"][fs:fs + n]; fhead[:n] = R["ego_heading"][fs:fs + n]; fval[:n] = 1.0
    # scene context and intent, from the resampled log at the log steps of these frames
    log_idx = R["steps"][t0:t0 + T]
    tl, kap, inter, found = scene_context(sd, log_idx)
    h = R["ego_heading"][t0:t0 + T].astype(np.float64)
    dh = np.degrees(np.sum(_wrap(np.diff(h[CONTEXT_RAW - 1:]))))
    intent = INTENT["left"] if dh > INTENT_DEG else INTENT["right"] if dh < -INTENT_DEG else INTENT["straight"]
    md = sd["metadata"]
    sl = slice(t0, t0 + T)
    out.update(
        ego_position=R["ego_position"][sl], ego_heading=R["ego_heading"][sl], ego_speed=R["ego_speed"][sl],
        has_lead=np.float32(target is not None), collision_flag=R["collision"][sl],
        episode_seed=np.int64(scen_index), scene_object_count=np.int64(len(set().union(*[set(p) for p in pf]))),
        clip_start_frame=np.int64(t0),
        future_ego_position=fpos, future_ego_heading=fhead, future_valid=fval,
        agents_track_id=a_id, agents_type=a_type, agents_position=a_pos, agents_heading_rel=a_head,
        agents_speed=a_speed, agents_bbox_2d=a_box, agents_visible=a_vis, agents_valid=a_valid,
        agents_dims=a_dims,
        tl_state=tl, lane_curvature=kap, in_intersection=inter, ego_lane_found=found,
        route_intent=np.int64(intent), route_dheading_deg=np.float32(dh),
        scenario_index=np.int64(scen_index), shard=np.int64(entry["shard"]), window_start=np.int64(t0),
        log_step=log_idx.astype(np.int64),
        log_ts_lo=np.asarray(md["log_ts_lo"])[log_idx].astype(np.float32),
        log_ts_hi=np.asarray(md["log_ts_hi"])[log_idx].astype(np.float32),
        frame_time=np.asarray(md["ts"])[log_idx].astype(np.float32),
        slice=np.int64(slice_code))
    return out


# ---------------------------------------------------------------- driver
_w = {}


def _init(db_dir, n, videos_path, ann_dir, offset=0):
    _w["env"] = make_env(db_dir, n)
    _w["offset"] = offset
    # positioned writes rather than a memmap: flushing a 36 GB mapping after
    # every scenario walks the whole mapping and dominated the train render
    _w["off"] = np.load(videos_path, mmap_mode="r").offset
    _w["fd"] = os.open(videos_path, os.O_WRONLY)
    _w["ann"] = ann_dir


def _one(job):
    from .stats import SLICE_CODE
    i, entry = job
    sd, R = rollout(_w["env"], i - _w.get("offset", 0))
    assert sd["id"] == entry["scenario_id"], (sd["id"], entry["scenario_id"])
    anns = []
    for j, (t0, s) in enumerate(zip(entry["starts"], entry["slices"])):
        assert t0 + CLIP <= R["video"].shape[1], "rollout shorter than window"
        clip = np.ascontiguousarray(R["video"][:, t0:t0 + CLIP], dtype=np.uint8)
        os.pwrite(_w["fd"], clip.tobytes(), _w["off"] + (2 * i + j) * clip.nbytes)
        anns.append(clip_labels(sd, R, t0, entry, i, SLICE_CODE[s]))
    stacked = {k: np.stack([a[k] for a in anns]) for k in anns[0]}
    tmp = os.path.join(_w["ann"], "%06d.tmp.npz" % i)
    np.savez(tmp, **stacked)
    os.replace(tmp, os.path.join(_w["ann"], "%06d.npz" % i))
    return i


def stage_render(sd_root, work, out_root, split, workers, limit=None):
    sel = json.load(open(os.path.join(work, "selection.json")))["selection"][split]
    n = len(sel) if limit is None else min(limit, len(sel))
    db = os.path.join(work, "db_" + split)
    out_dir = os.path.join(out_root, split)
    ann_dir = os.path.join(work, "ann_" + split)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    vp = os.path.join(out_dir, "videos.npy")
    if not os.path.exists(vp):
        mm = np.lib.format.open_memmap(vp, mode="w+", dtype=np.uint8, shape=(2 * len(sel), 3, CLIP, IMG, IMG))
        mm.flush()
        del mm
    done = {int(f[:6]) for f in os.listdir(ann_dir) if f.endswith(".npz") and ".tmp" not in f}
    jobs = [(i, sel[i]) for i in range(n) if i not in done]
    print("%s: %d scenarios, %d done, %d to render, %d workers" % (split, n, len(done), len(jobs), workers), flush=True)
    # MetaDrive reloads the whole dataset summary on every worker start, so a
    # large split renders from chunked databases of CHUNK scenarios each.
    from .select import write_db
    t0 = time.time()
    k = 0
    for c0 in range(0, n, CHUNK):
        cj = [j for j in jobs if c0 <= j[0] < c0 + CHUNK]
        if not cj:
            continue
        cdb = db
        if len(sel) > CHUNK:
            cdb = "%s_c%02d" % (db, c0 // CHUNK)
            write_db(sel[c0:c0 + CHUNK], sd_root, cdb)
        m = min(CHUNK, len(sel) - c0) if len(sel) > CHUNK else len(sel)
        off = c0 if len(sel) > CHUNK else 0
        with Pool(workers, initializer=_init, initargs=(cdb, m, vp, ann_dir, off), maxtasksperchild=100) as pool:
            for _ in pool.imap_unordered(_one, cj, chunksize=1):
                k += 1
                if k % 500 == 0 or k == len(jobs):
                    el = time.time() - t0
                    print("  %s %d/%d  %.2f scen/s  ETA %.2f h" % (split, k, len(jobs), k / el,
                          (len(jobs) - k) / (k / el) / 3600), flush=True)
