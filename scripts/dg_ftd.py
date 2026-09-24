"""Fréchet Trajectory Distance, and the floor it sits on.

FTD encodes each clip with MTR's `agent_polyline_encoder` over 11-frame windows
at stride 10, averages the ten window features into one vector per clip, and
takes a Fréchet distance between the estimated set and the real set.

Two things are measured. The first is the distortion each extractor
configuration introduces: every clip here is real footage, so a non-zero FTD is
the benchmark's own noise. The second is the harder question, which their paper
does not address: a Fréchet distance between two finite samples of the *same*
distribution is not zero, so splitting the real trajectories in half and scoring
one against the other gives the floor below which no FTD difference means
anything at this sample size.

Run from the DrivingGen checkout: `parse_config` reads its config and checkpoint
by relative path.
"""
from __future__ import annotations

import argparse
import json
import os
import pdb
import sys

import numpy as np

REPO = "third_party/DrivingGen"
# MTR is not vendored; scripts/setup_third_party.sh fetches it here
MTR = os.environ.get("MTR_DIR", os.path.join(REPO, "third_parties", "MTR"))
def cells(geom: str):
    return [
        (geom, "predicted", "top + predicted (as published)"),
        (geom, "calibrated_slam", "top + calibrated_slam"),
        (geom, "calibrated_both", "top + calibrated_both"),
        ("full_" + geom, "predicted", "full + predicted"),
        ("full_" + geom, "calibrated_both", "full + calibrated_both (repaired)"),
    ]


def traj_dirs(geom: str):
    return {geom: "/root/driving-gen/traj/%s" % geom,
            "full_" + geom: "/root/driving-gen/traj_roi/full_%s" % geom}


def ensure_intention_points():
    """MTR's constructor loads clustered intention points for its motion decoder.

    FTD uses only `context_encoder.agent_polyline_encoder`, the decoder is never
    run, and the points are a plain dict rather than a state-dict entry, so
    nothing loaded from the checkpoint depends on them. The file is absent from
    the vendored copy, so a placeholder of the right shape is written to let the
    model construct.
    """
    import pickle
    import yaml

    cfg_path = os.path.join(MTR, "tools", "cfgs",
                            "waymo", "mtr+100_percent_data.yaml")
    with open(cfg_path) as f:
        y = yaml.safe_load(f)
    md = y["MODEL"]["MOTION_DECODER"]
    rel = md["INTENTION_POINTS_FILE"]
    modes = int(md.get("NUM_MOTION_MODES", 64))
    types = md.get("OBJECT_TYPE") or y["MODEL"].get("OBJECT_TYPE") \
        or ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]

    dst = os.path.join(MTR, rel)
    if os.path.exists(dst):
        print("intention points: already present")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob = {t: np.zeros((modes, 2), np.float32) for t in types}
    blob.update({t: np.zeros((modes, 2), np.float32)
                 for t in ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")})
    with open(dst, "wb") as f:
        pickle.dump(blob, f)
    print("intention points: wrote placeholder %s, %d modes, types %s"
          % (rel, modes, list(blob)))


def load_dg():
    """Their trajs package, with the interactive debugger on NaN disabled."""
    for p in (os.path.join(REPO, "drivinggen"),
              MTR):
        if p not in sys.path:
            sys.path.insert(0, p)
    pdb.set_trace = lambda *a, **k: None          # their NaN branch drops into pdb
    ensure_intention_points()
    from trajs import traj_distribution as td
    return td


def build(clips, traj_dir, names, repo):
    """Aligned, smoothed predictions and the matching ground truth, exactly as
    their sampler prepares them before FTD."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cond_eval.dg_eval import score as S
    preds, gts, used = [], [], []
    for n in names:
        f = os.path.join(traj_dir, n + ".npz")
        if not os.path.exists(f):
            continue
        with open(os.path.join(clips, n, "meta.json")) as fh:
            m = json.load(fh)
        d = np.load(f, allow_pickle=True)
        p, g, _ = S.prepare_pair(repo, d["locs"], np.asarray(m["ego_xy"], float),
                                 with_scale=False)
        preds.append(np.asarray(p, float))
        gts.append(np.asarray(g, float))
        used.append(n)
    return preds, gts, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--percl", default="out/dg_calibration/"
                                       "per_clip_dg_predicted_ego_fixed_scale.npz")
    ap.add_argument("--out", default="out/dg_calibration/ftd.json")
    ap.add_argument("--feats", default="/root/driving-gen/viz/ftd_feats.npz")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--geom", default="dg", choices=["dg", "stage2"])
    args = ap.parse_args()
    CELLS, TRAJ = cells(args.geom), traj_dirs(args.geom)

    ref = np.load(args.percl, allow_pickle=True)
    names = [str(x) for x in ref["names"]]
    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            p = np.asarray(json.load(f)["ego_xy"], float)
        driven.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    moving = [n for n, d in zip(names, driven) if d >= 10]
    print("moving clips: %d" % len(moving), flush=True)

    td = load_dg()
    feats, report = {}, {"n_clips": len(moving), "stride": args.stride, "ftd": {}}

    for tag, mode, label in CELLS:
        traj_dir = os.path.join(TRAJ[tag], mode)
        if not os.path.isdir(traj_dir):
            print("skip %s" % label); continue
        preds, gts, used = build(args.clips, traj_dir, moving, args.repo)
        print("\n%s: %d clips" % (label, len(used)), flush=True)
        Fp, Fg = td.infer(preds, gts, stride=args.stride)
        Fp, Fg = np.asarray(Fp), np.asarray(Fg)
        ftd = float(td.compute_fid_feats(Fp, Fg))
        print("  FTD = %.4f   (features %s)" % (ftd, Fp.shape), flush=True)
        report["ftd"][label] = ftd
        feats["pred_%s_%s" % (tag, mode)] = Fp
        feats["gt"] = Fg

    # the floor: two disjoint halves of the real set, scored against each other
    Fg = feats["gt"]
    rng = np.random.default_rng(2026)
    print("\n=== FTD between two halves of the real set ===", flush=True)
    report["real_vs_real"] = {}
    for n in (25, 50, 75, len(Fg) // 2):
        if 2 * n > len(Fg):
            continue
        vals = []
        for _ in range(20):
            idx = rng.permutation(len(Fg))
            vals.append(float(td.compute_fid_feats(Fg[idx[:n]], Fg[idx[n:2 * n]])))
        v = np.asarray(vals)
        report["real_vs_real"]["n=%d" % n] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p95": float(np.percentile(v, 95))}
        print("  n=%3d per side: FTD %.4f +- %.4f   p95 %.4f"
              % (n, v.mean(), v.std(ddof=1), np.percentile(v, 95)), flush=True)

    np.savez_compressed(args.feats, **feats)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote %s and %s" % (args.out, args.feats))


if __name__ == "__main__":
    main()
