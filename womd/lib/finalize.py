"""Stack per-scenario labels into the DiskDataset layout and run the section 8 checks.

Per split under OUT/<split>/: videos.npy (written by render), annotations.npz,
meta.json, pixel_stats.json, scenarios.json (scenario ids; clip 2 i + j belongs to
scenario i). Runs in the training env (torch + f_toy).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "third_party/Task_specific_JDM")

from .render import CLIP, HORIZON, IMG, N_AGENTS, CAMERA_POS, CAMERA_HPR, CAMERA_FOV, \
    TARGET_MAX_DIST, TARGET_CONE_DEG, INTENT, TL_CODE
from .stats import SLICES, STATIC_M


def stack(work, out_root, split):
    sel = json.load(open(os.path.join(work, "selection.json")))
    entries = sel["selection"][split]
    ann_dir = os.path.join(work, "ann_" + split)
    n = len(entries)
    missing = [i for i in range(n) if not os.path.exists(os.path.join(ann_dir, "%06d.npz" % i))]
    assert not missing, "%d scenarios not rendered, first %s" % (len(missing), missing[:5])
    first = np.load(os.path.join(ann_dir, "%06d.npz" % 0))
    out = {k: np.empty((2 * n,) + first[k].shape[1:], first[k].dtype) for k in first.files}
    for i in range(n):
        a = np.load(os.path.join(ann_dir, "%06d.npz" % i))
        for k in a.files:
            out[k][2 * i:2 * i + 2] = a[k]
    d = os.path.join(out_root, split)
    np.savez(os.path.join(d, "annotations.npz"), **out)
    json.dump([dict(scenario_id=e["scenario_id"], shard=e["shard"], file=e["file"], starts=e["starts"],
                    slices=e["slices"]) for e in entries], open(os.path.join(d, "scenarios.json"), "w"))
    meta = dict(
        num_samples=2 * n, frames=CLIP, img_size=IMG, fps=7, seed=None,
        camera_pos=list(CAMERA_POS), camera_hpr=list(CAMERA_HPR), camera_fov=CAMERA_FOV,
        target_max_dist=TARGET_MAX_DIST, target_forward_cone_deg=TARGET_CONE_DEG,
        data_directory="/opt/womd/work/db_" + split, source="waymo_open_motion_v1.3.0 training_20s",
        split=split, scenarios_used=n, clips_per_scenario=2,
        waypoint_horizon=HORIZON, n_agents=N_AGENTS,
        target_type_codes=dict(vehicle=0, pedestrian=1, cyclist=2, none=-1),
        slice_codes={s: i for i, s in enumerate(SLICES)}, route_intent_codes=INTENT,
        tl_state_codes=dict(none=-1, unknown=0, stop=1, caution=2, go=3),
        selection_report=sel["report"][split], channel_order="rgb")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return out, meta


def pixel_stats(out_root, split):
    from f_toy.datasets.waymo.disk import DiskDataset
    from f_toy.data.pixel_stats import pixel_stats as ps
    return ps(DiskDataset(os.path.join(out_root, split)))


def checks(out_root, splits):
    rep = {}
    ids = {}
    for sp in splits:
        d = os.path.join(out_root, sp)
        a = np.load(os.path.join(d, "annotations.npz"))
        sc = json.load(open(os.path.join(d, "scenarios.json")))
        ids[sp] = set(s["scenario_id"] for s in sc)
        r = {}
        ft = a["frame_time"]
        dt = np.diff(ft, axis=1)
        r["frame_dt_min_max"] = [float(dt.min()), float(dt.max())]
        r["on_grid_max_err_s"] = float(np.abs(ft * 7 - np.round(ft * 7)).max() / 7)
        r["log_bracket_max_gap_s"] = float((a["log_ts_hi"] - a["log_ts_lo"]).max())
        p = a["ego_position"]
        step = np.hypot(*np.moveaxis(np.diff(p, axis=1), -1, 0))
        v_diff = step * 7
        sp_log = 0.5 * (a["ego_speed"][:, 1:] + a["ego_speed"][:, :-1])
        err = np.abs(v_diff - sp_log)
        r["speed_diff_vs_logged_mps"] = dict(median=float(np.median(err)), p99=float(np.percentile(err, 99)),
                                             max=float(err.max()))
        r["max_ego_step_m"] = float(step.max())
        r["heading_jump_max_deg"] = float(np.degrees(np.abs((np.diff(a["ego_heading"], axis=1) + np.pi) % (2 * np.pi) - np.pi)).max())
        path = step.sum(1)
        r["static_frac"] = float((path < STATIC_M).mean())
        r["slices"] = {s: int((a["slice"] == i).sum()) for i, s in enumerate(SLICES)}
        r["route_intent"] = {k: int((a["route_intent"] == v).sum()) for k, v in INTENT.items()}
        r["ego_lane_found_frac"] = float(a["ego_lane_found"].mean())
        r["has_lead_frac"] = float(a["has_lead"].mean())
        r["future_valid_frac"] = float(a["future_valid"].mean())
        r["agents_valid_mean"] = float(a["agents_valid"].sum(-1).mean())
        rep[sp] = r
    names = list(ids)
    rep["disjoint"] = {"%s-%s" % (x, y): len(ids[x] & ids[y]) for i, x in enumerate(names) for y in names[i + 1:]}
    return rep


def stage_finalize(work, out_root, split=None):
    splits = [split] if split else ["train", "val", "test"]
    for sp in splits:
        stack(work, out_root, sp)
        print(sp, "pixel_stats", pixel_stats(out_root, sp), flush=True)
    rep = checks(out_root, splits)
    json.dump(rep, open(os.path.join(out_root, "checks.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))
