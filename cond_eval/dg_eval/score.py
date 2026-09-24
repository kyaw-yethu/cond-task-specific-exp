"""Scoring, using DrivingGen's own alignment and metric functions.

Nothing here reimplements a metric. `gt_2_ego`, `ego_y_2_x`,
`slam_align_to_gt_fix_origin` and `smooth_traj_sg` are loaded out of their
`z-sample_ftd.py` (whose hyphen makes it unimportable by name, so it is loaded
by path), and ADE / FDE / DTW / Hausdorff / success rate / comfort / consistency
come from their `trajs` package.

Two knobs their script fixes as constants are exposed, because both are
load-bearing on nuScenes:

`with_scale`  their pipeline aligns with `with_scale=False`, so nothing is ever
              rescaled to the reference and ADE is directly exposed to
              UniDepth's absolute-scale error. Running both settings splits the
              floor into a scale part and a shape part.

`dt`          0.1 in their code. Correct for the 10 Hz `dg` geometry; the
              stage-2 geometry is 12 Hz, and every derivative-based metric is
              off by a constant factor if that is not changed.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import numpy as np

_DG = None


def _ftd_unavailable(*a, **k):
    raise RuntimeError("FTD needs MTR's compiled CUDA ops; score it separately")


def load_drivinggen(repo: str):
    """Import their helpers once. Returns a namespace of the functions used."""
    global _DG
    if _DG is not None:
        return _DG
    pkg = os.path.join(repo, "drivinggen")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)

    # z-sample_ftd.py pulls in get_ftd at module level, which reaches MTR's
    # compiled CUDA attention op. FTD is scored separately; standing a stub in
    # for that one import lets their four alignment helpers load unchanged.
    if "trajs.traj_distribution" not in sys.modules:
        try:
            importlib.import_module("trajs.traj_distribution")
        except Exception:
            stub = types.ModuleType("trajs.traj_distribution")
            stub.get_ftd = _ftd_unavailable
            sys.modules["trajs.traj_distribution"] = stub

    spec = importlib.util.spec_from_file_location(
        "dg_sample_ftd", os.path.join(pkg, "z-sample_ftd.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dg_sample_ftd"] = mod
    spec.loader.exec_module(mod)          # __main__ guard keeps the script body out

    from trajs import traj_alignment
    from trajs import traj_quality
    from trajs import traj_consistency

    ns = types.SimpleNamespace(
        gt_2_ego=mod.gt_2_ego,
        ego_y_2_x=mod.ego_y_2_x,
        slam_align=mod.slam_align_to_gt_fix_origin,
        smooth=mod.smooth_traj_sg,
        align=traj_alignment,
        quality=traj_quality,
        consistency=traj_consistency,
    )
    _DG = ns
    return _DG


def prepare_pair(repo: str, locs: np.ndarray, gt_xy: np.ndarray, *,
                 with_scale: bool = False, dt: float = 0.1,
                 win_sec: float = 0.4):
    """Their exact sequence: GT into the first-frame ego frame, prediction into
    their axis convention, Procrustes onto the GT with the origin pinned, then
    Savitzky-Golay smoothing of both."""
    dg = load_drivinggen(repo)
    gt_local_xy, _, _ = dg.gt_2_ego(np.asarray(gt_xy, float))
    pred_xy = dg.ego_y_2_x(np.asarray(locs, float))
    pred_xy, s, R = dg.slam_align(pred_xy, gt_local_xy, with_scale=with_scale)
    pred_xy = dg.smooth(pred_xy, dt=dt, win_sec=win_sec, poly=3)
    gt_local_xy = dg.smooth(gt_local_xy, dt=dt, win_sec=win_sec, poly=3)
    return np.asarray(pred_xy, float), np.asarray(gt_local_xy, float), float(s)


def score_pairs(repo: str, preds: np.ndarray, gts: np.ndarray, *, dt: float = 0.1):
    """Per-clip values for every alignment and quality metric in their suite
    that does not need the MTR encoder."""
    dg = load_drivinggen(repo)
    a = dg.align
    out = {
        "ade": np.asarray(a.ade(preds, gts, reduce="none"), float),
        "fde": np.asarray(a.fde(preds, gts, reduce="none"), float),
        "dtw": np.asarray(a.dtw(preds, gts, reduce="none"), float),
        "hausdorff": np.asarray(a.hausdorff(preds, gts, reduce="none"), float),
        "success_rate": np.asarray(a.success_rate(preds, gts, threshold=3.0,
                                                  reduce="none"), float),
        "dynamic_consistency": np.asarray(
            a.dynamic_consistency(preds, gts, dt=dt, reduce="none"), float),
    }
    q = dg.quality.comfort_score_norm(preds, dt=dt, reduce="none")
    out["comfort"] = np.asarray(q, float)
    out["traj_consistency"] = np.asarray(
        dg.consistency.trajectory_consistency(preds, dt=dt, reduce="none"), float)
    return out


def summarise(per_clip: dict) -> dict:
    """Mean, standard error and n over the clips that produced a value."""
    out = {}
    for k, v in per_clip.items():
        v = np.asarray(v, float)
        good = np.isfinite(v)
        n = int(good.sum())
        out[k] = {
            "mean": float(v[good].mean()) if n else float("nan"),
            "sem": float(v[good].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n": n,
        }
    return out
