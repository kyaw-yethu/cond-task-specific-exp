"""Camera-projection overlays for the six clips in the bird's-eye figure.

`plot_dg_traj.py` picks six clips spanning the error distribution and draws them
from above. The same six read differently projected back onto the road they
describe, where a metre of error is a lane rather than a number, so this renders
that view for each of them: one grid for comparison and six standalone frames.

Selection is imported rather than repeated, so the panels here are the same
clips in the same order as the bird's-eye figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import plot_dg_traj as P
from cond_eval.dg_eval import nusc_index as NI
from cond_eval.dg_eval import score as S


def camera_of(index, raw_dir, scene):
    cs_raw = json.load(open(os.path.join(raw_dir, "calibrated_sensor.json")))
    cs = next(c for c in cs_raw if c["token"] == index[scene].cs_token)
    return np.asarray(cs["translation"], float), NI.quat_to_rot(cs["rotation"])


def draw_overlay(ax, args, name, index, raw_dir, label=None, ticks=True):
    meta, ref, fixed, scaled, s = P.load_clip(args.clips, args.traj, name, args.repo)
    cam_t, cam_R = camera_of(index, raw_dir, name)
    K = np.asarray(meta["K"], float)

    img = np.array(Image.open(os.path.join(args.clips, name, "images", "00000.jpg")))
    h, w = img.shape[:2]
    base = img.astype(np.float32)
    base = base * .70 + 52 if base.mean() > 110 else np.clip(base * 1.5 + 18, 0, 255)
    ax.imshow(base.astype(np.uint8))

    P.ribbon(ax, scaled, K, cam_t, cam_R, h, w, P.PR_C, .55, 1.4, ls="--")
    c_gt = P.ribbon(ax, ref, K, cam_t, cam_R, h, w, P.GT_C, .95, 2.6)
    c_pr = P.ribbon(ax, fixed, K, cam_t, cam_R, h, w, P.PR_C, .95, 2.2)

    if ticks:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
        for metres in (10, 20, 30, 40, 50, 60, 70):
            if metres > d[-1]:
                break
            i = int(np.searchsorted(d, metres))
            uv, _ = P.project(ref[i:i + 1], K, cam_t, cam_R, h, w)
            if len(uv) and 0 < uv[0, 0] < w - 26 and 0 < uv[0, 1] < h - 4:
                ax.plot(uv[0, 0], uv[0, 1], "o", color=P.GT_C, ms=3, zorder=6)
                ax.text(uv[0, 0] + 5, uv[0, 1], "%d" % metres, color="white",
                        fontsize=5.5, va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.13", fc=P.GT_C,
                                  ec="none", alpha=.9))
    for c, col in ((c_gt, P.GT_C), (c_pr, P.PR_C)):
        if len(c) and 0 < c[-1, 0] < w and 0 < c[-1, 1] < h:
            ax.plot(c[-1, 0], c[-1, 1], "o", color=col, ms=5, zorder=7,
                    markeredgecolor="white", markeredgewidth=1)

    dg = S.load_drivinggen(args.repo)
    ade = float(dg.align.ade(fixed[None], ref[None]).ravel()[0])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    title = "%s  %s" % (label, name) if label else name
    ax.set_title("%s   ADE %.2f m   %.0f m driven" % (title, ade, P.path_len(ref)),
                 fontsize=8.5, color=P.INK, loc="left", pad=4)
    return ade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/stage2")
    ap.add_argument("--traj", default="/root/driving-gen/traj_roi/full_stage2/calibrated_both")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--raw", default="/root/driving-gen/nuscenes/raw/v1.0-trainval")
    ap.add_argument("--index", default="/root/driving-gen/nuscenes/index.json")
    ap.add_argument("--percl", default="out/dg_calibration/"
                                       "per_clip_full_stage2_calibrated_both_ego_fixed_scale.npz")
    ap.add_argument("--outdir", default="out/dg_calibration")
    ap.add_argument("--suffix", default="_s2_repaired")
    args = ap.parse_args()

    d = np.load(args.percl, allow_pickle=True)
    ade, names = d["ade"], [str(x) for x in d["names"]]
    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            driven.append(P.path_len(np.asarray(json.load(f)["ego_xy"], float)))
    moving = np.asarray(driven) >= P.MOVING_M
    m_idx = np.where(moving)[0]
    order = m_idx[np.argsort(ade[m_idx])]
    pick = [("best", 0), ("p25", int(.25 * len(order))), ("median", len(order) // 2),
            ("p75", int(.75 * len(order))), ("p95", int(.95 * len(order))),
            ("worst", len(order) - 1)]
    chosen = [(lab, names[order[i]]) for lab, i in pick]
    print("six clips, same selection as the bird's-eye figure:")
    for lab, n in chosen:
        print("  %-7s %-14s ADE %6.2f" % (lab, n, ade[names.index(n)]))

    index = NI.load_index(args.index)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 5.9), dpi=190)
    for ax, (lab, name) in zip(axes.ravel(), chosen):
        draw_overlay(ax, args, name, index, args.raw, label=lab)
    handles = [plt.Line2D([], [], color=P.GT_C, lw=2.6, label="true ego path"),
               plt.Line2D([], [], color=P.PR_C, lw=2.2, label="estimated"),
               plt.Line2D([], [], color=P.PR_C, lw=1.4, ls="--", alpha=.6,
                          label="estimated, scale freed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Both paths projected onto the first frame, the same six clips",
                 fontsize=11.5, color=P.INK, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.962), w_pad=0.4, h_pad=1.2)
    grid = os.path.join(args.outdir, "traj_overlay_grid%s.png" % args.suffix)
    fig.savefig(grid, facecolor="white")
    plt.close(fig)
    print("\nwrote", grid)

    for lab, name in chosen:
        f, ax = plt.subplots(figsize=(7.4, 4.4), dpi=190)
        draw_overlay(ax, args, name, index, args.raw, label=lab)
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  framealpha=.9, facecolor="white", edgecolor="none")
        f.tight_layout()
        out = os.path.join(args.outdir,
                           "overlay%s_%s_%s.png" % (args.suffix, lab, name))
        f.savefig(out, facecolor="white", bbox_inches="tight")
        plt.close(f)
        print("wrote", out)


if __name__ == "__main__":
    main()
