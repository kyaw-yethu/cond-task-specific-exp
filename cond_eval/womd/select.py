"""Choose scenarios and window pairs for each split (DATASET_PLAN.md section 4).

train / val: the pair ({0,66} or {33,99}) with more ego path length; static clips
capped at 3% of the split; scenarios taken in a seeded shuffle until the target.
test: pairs chosen to fill the slice quotas, scarcest slice first.
Writes <work>/selection.json and one ScenarioNet database per split.
"""
import collections
import json
import os
import pickle

import numpy as np

from .stats import PAIRS, SLICES

TARGET = dict(train=20000, val=2000, test=2000)
STATIC_FRAC = 0.03
QUOTA = dict(turn=800, lead_closing=700, braking=600, vru_nearby=500, cruising=1200, static=200)
SEED = dict(train=0, val=1, test=2)


def _rows(sd_root, shards):
    rows = []
    for s in shards:
        rows += json.load(open(os.path.join(sd_root, "%05d" % s, "stats.json")))
    return rows


def _pairs(r):
    out = []
    for pa in PAIRS:
        ws = [r["windows"][str(s)] for s in pa]
        if all(w is not None for w in ws):
            out.append((pa, ws))
    return out


def _entry(r, pa, ws):
    return dict(file=r["file"], shard=r["shard"], scenario_id=r["scenario_id"], length=r["length"],
                starts=list(pa), slices=[w["slice"] for w in ws], windows=ws)


def select_trainval(rows, split):
    n = TARGET[split]
    cap = int(round(STATIC_FRAC * 2 * n))
    rng = np.random.RandomState(SEED[split])
    order = rng.permutation(len(rows))
    out, n_static, skipped = [], 0, collections.Counter()
    for i in order:
        r = rows[i]
        ps = _pairs(r)
        if not ps:
            skipped["short"] += 1
            continue
        pa, ws = max(ps, key=lambda p: sum(w["path_m"] for w in p[1]))
        k = sum(w["static"] for w in ws)
        if n_static + k > cap:
            skipped["static_cap"] += 1
            continue
        n_static += k
        out.append(_entry(r, pa, ws))
        if len(out) == n:
            break
    return out, dict(skipped=dict(skipped), static_clips=n_static, static_cap=cap)


def select_test(rows):
    rng = np.random.RandomState(SEED["test"])
    rows = [rows[i] for i in rng.permutation(len(rows))]
    avail = collections.Counter(w["slice"] for r in rows for _, ws in _pairs(r) for w in ws)
    need = dict(QUOTA)
    order = sorted(QUOTA, key=lambda s: avail[s] / QUOTA[s])      # scarcest first
    taken, out = set(), []

    def fits(ws):
        c = collections.Counter(w["slice"] for w in ws)
        return all(need[s] >= k for s, k in c.items())

    for s in order:
        for idx, r in enumerate(rows):
            if need[s] == 0 or len(out) == TARGET["test"]:
                break
            if idx in taken:
                continue
            cands = [(pa, ws) for pa, ws in _pairs(r) if s in [w["slice"] for w in ws] and fits(ws)]
            if not cands:
                continue
            pa, ws = max(cands, key=lambda p: sum(w["slice"] == s for w in p[1]))
            for w in ws:
                need[w["slice"]] -= 1
            taken.add(idx)
            out.append(_entry(r, pa, ws))
    got = collections.Counter(sl for e in out for sl in e["slices"])
    return out, dict(quota=QUOTA, got=dict(got), available_windows=dict(avail), order=order)


def write_db(entries, sd_root, db_dir):
    os.makedirs(db_dir, exist_ok=True)
    summ, mapping, cache = {}, {}, {}
    for e in entries:
        sh = e["shard"]
        if sh not in cache:
            cache[sh] = pickle.load(open(os.path.join(sd_root, "%05d" % sh, "summary.pkl"), "rb"))
        summ[e["file"]] = cache[sh][e["file"]]
        mapping[e["file"]] = os.path.join(sd_root, "%05d" % sh)
    pickle.dump(summ, open(os.path.join(db_dir, "dataset_summary.pkl"), "wb"))
    pickle.dump(mapping, open(os.path.join(db_dir, "dataset_mapping.pkl"), "wb"))


def stage_select(sd_root, work, shards):
    os.makedirs(work, exist_ok=True)
    sel, report = {}, {}
    for split in ("train", "val", "test"):
        rows = _rows(sd_root, shards[split])
        if split == "test":
            sel[split], rep = select_test(rows)
        else:
            sel[split], rep = select_trainval(rows, split)
        rep.update(pool=len(rows), scenarios=len(sel[split]),
                   slices=dict(collections.Counter(s for e in sel[split] for s in e["slices"])))
        report[split] = rep
        write_db(sel[split], sd_root, os.path.join(work, "db_" + split))
        print(split, json.dumps(rep), flush=True)
    ids = {k: set(e["scenario_id"] for e in v) for k, v in sel.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    for k, v in sel.items():
        assert len(ids[k]) == len(v), "duplicate scenario id in " + k
    json.dump(dict(selection=sel, report=report), open(os.path.join(work, "selection.json"), "w"))
