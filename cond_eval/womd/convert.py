"""One WOMD tfrecord shard -> resampled ScenarioNet pickles + per-window stats.

Output per shard: <out>/<shard>/sd_waymo_v1.3_<id>.pkl and <out>/<shard>/stats.json.
"""
import json
import os
import pickle

import numpy as np

from .resample import resample_sd, HZ
from .stats import STARTS, window_stats, slice_of
from .tfrecord import records


def convert_shard(tfrecord_path, out_root, shard):
    from scenarionet.converter.waymo.utils import convert_waymo_scenario
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2
    from metadrive.scenario.scenario_description import ScenarioDescription as SD

    out_dir = os.path.join(out_root, "%05d" % shard)
    stats_path = os.path.join(out_dir, "stats.json")
    if os.path.exists(stats_path):
        return json.load(open(stats_path))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rec in records(tfrecord_path):
        s = scenario_pb2.Scenario()
        s.ParseFromString(rec)
        sd = convert_waymo_scenario(s, "v1.3")
        raw_len, raw_ts = sd["length"], np.asarray(sd["metadata"]["ts"], dtype=np.float64)
        sd = resample_sd(sd, HZ)
        if hasattr(SD, "update_summaries"):
            SD.update_summaries(sd)
        fname = SD.get_export_file_name("waymo", "v1.3", sd["id"])
        d = sd.to_dict() if hasattr(sd, "to_dict") else dict(sd)
        SD.sanity_check(d, check_self_type=True)
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(d, f)
        wins = {}
        for st in STARTS:
            w = window_stats(sd, st)
            if w is not None:
                w["slice"] = slice_of(w)
            wins[str(st)] = w
        rows.append(dict(
            file=fname, shard=shard, scenario_id=sd["id"], length=int(sd["length"]),
            raw_length=int(raw_len), raw_duration_s=float(raw_ts[-1] - raw_ts[0]),
            max_gap_s=float(np.diff(raw_ts).max()), windows=wins,
            metadata=d["metadata"]))
    meta = {r["file"]: r.pop("metadata") for r in rows}
    with open(os.path.join(out_dir, "summary.pkl"), "wb") as f:
        pickle.dump(meta, f)
    json.dump(rows, open(stats_path + ".tmp", "w"))
    os.replace(stats_path + ".tmp", stats_path)
    return rows
