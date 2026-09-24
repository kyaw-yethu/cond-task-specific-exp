"""Trajectory figures for the calibration report.

Two views of the same fact. The bird's-eye grid puts estimated and true paths
side by side across the error distribution, which is where the shape-versus-
length split is legible. The camera overlay projects both paths back onto the
road they describe, which is where a metre of error becomes something you can
see.

DrivingGen's scoring frame, after `gt_2_ego`, has the first frame at the origin
with the initial heading along +x, so column 0 is forward and column 1 is left.
That is the nuScenes ego convention too, which is what makes the projection in
`make_overlay` a direct composition with the CAM_FRONT extrinsic.

Panels are drawn from clips that actually move. A parked car scores ADE 0.00 by
definition and would otherwise occupy the "best" slot while showing nothing;
those clips are counted in the caption instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image

from cond_eval.dg_eval import nusc_index as NI
from cond_eval.dg_eval import score as S

GT_C = "#0d7a6e"
PR_C = "#c2410c"
INK = "#1a1a1a"
GREY = "#6b7280"
FAINT = "#dfe4e9"
MIN_HALF_W = 11.0          # metres, so a straight clip is not drawn as a sliver
MOVING_M = 10.0            # metres of true path required to be a usable panel


def path_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def load_clip(clips_dir, traj_dir, name, repo):
    with open(os.path.join(clips_dir, name, "meta.json")) as f:
        meta = json.load(f)
    d = np.load(os.path.join(traj_dir, name + ".npz"), allow_pickle=True)
    gt = np.asarray(meta["ego_xy"], float)
    fixed, ref, _ = S.prepare_pair(repo, d["locs"], gt, with_scale=False)
    scaled, _, s = S.prepare_pair(repo, d["locs"], gt, with_scale=True)
    return meta, ref, fixed, scaled, float(s)


def bev_panel(ax, ref, fixed, scaled, title, sub):
    # x is lateral (positive right), y is forward
    ax.plot(-scaled[:, 1], scaled[:, 0], "--", color=PR_C, lw=1.6, alpha=.55,
            zorder=2, label="estimated, scale freed")
    ax.plot(-ref[:, 1], ref[:, 0], "-", color=GT_C, lw=3.0, zorder=3,
            solid_capstyle="round", label="true ego path")
    ax.plot(-fixed[:, 1], fixed[:, 0], "-", color=PR_C, lw=2.2, zorder=4,
            solid_capstyle="round", label="estimated")
    ax.plot([0], [0], "o", color=INK, ms=5.5, zorder=6)
    ax.plot(-ref[-1, 1], ref[-1, 0], "o", color=GT_C, ms=5.5, zorder=6)
    ax.plot(-fixed[-1, 1], fixed[-1, 0], "o", color=PR_C, ms=5.5, zorder=6)

    pts = np.vstack([ref, fixed, scaled])
    cx = float(np.median(-pts[:, 1]))
    x0, x1 = (-pts[:, 1]).min(), (-pts[:, 1]).max()
    half = max(MIN_HALF_W, (x1 - x0) / 2 * 1.15)
    ax.set_xlim(cx - half, cx + half)
    y0, y1 = pts[:, 0].min(), pts[:, 0].max()
    padded = max(6.0, (y1 - y0) * .08)
    ax.set_ylim(y0 - padded, y1 + padded)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=10, color=INK, pad=6, loc="left")
    ax.text(0.03, 0.97, sub, transform=ax.transAxes, fontsize=8.2,
            color=GREY, va="top", family="monospace", zorder=8,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="none", alpha=.82))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0d5da")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(labelsize=7.5, colors=GREY, length=3)
    ax.grid(True, color="#f0f2f5", lw=.9, zorder=0)
    ax.set_axisbelow(True)


def make_bev(args, chosen, n_static, out):
    repo = args.repo
    dg = S.load_drivinggen(repo)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), dpi=170)
    for ax, (label, name) in zip(axes.ravel(), chosen):
        meta, ref, fixed, scaled, s = load_clip(args.clips, args.traj, name, repo)
        ade = float(dg.align.ade(fixed[None], ref[None]).ravel()[0])
        ade_s = float(dg.align.ade(scaled[None], ref[None]).ravel()[0])
        bev_panel(ax, ref, fixed, scaled, "%s   %s" % (label, name),
                  "ADE       %6.2f m\nscale-free%6.2f m\nfit scale %6.3f\ndriven    %6.1f m"
                  % (ade, ade_s, s, path_len(ref)))
    for ax in axes[1]:
        ax.set_xlabel("lateral, m", fontsize=8.5, color=GREY)
    for ax in axes[:, 0]:
        ax.set_ylabel("forward, m", fontsize=8.5, color=GREY)

    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index("true ego path"), l.index("estimated"),
             l.index("estimated, scale freed")]
    fig.legend([h[i] for i in order], [l[i] for i in order],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.003))
    fig.suptitle("DrivingGen's extractor on real nuScenes clips, bird's-eye view",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.988)
    fig.text(0.012, 0.958,
             "Six of the 250 clips, spanning the error distribution. Equal axes, every "
             "panel at least %.0f m wide.\n%d of the 250 never move more than %.0f m and "
             "are excluded here: a parked car scores ADE near zero whatever the "
             "estimator does." % (2 * MIN_HALF_W, n_static, MOVING_M),
             fontsize=8.5, color=GREY, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def project(pts_fwd_left, K, cam_t, cam_R, h, w, z=0.0):
    p = np.zeros((len(pts_fwd_left), 3))
    p[:, 0] = pts_fwd_left[:, 0]
    p[:, 1] = pts_fwd_left[:, 1]
    p[:, 2] = z
    cam = (cam_R.T @ (p - cam_t).T).T
    ok = cam[:, 2] > 1.0        # in front of the camera, not behind or on it
    if not ok.any():
        return np.zeros((0, 2)), ok
    uv = (K @ cam[ok].T).T
    return uv[:, :2] / uv[:, 2:3], ok


def ribbon(ax, pts, K, cam_t, cam_R, h, w, color, alpha, lw, ls="-"):
    """Draw the path as a vehicle-width band so it reads as a driven track."""
    half = np.zeros_like(pts)
    half[:, 1] = 0.9                       # 1.8 m track, offset in +left
    left, okl = project(pts + half, K, cam_t, cam_R, h, w)
    right, okr = project(pts - half, K, cam_t, cam_R, h, w)
    n = min(len(left), len(right))
    if n > 2:
        poly = np.vstack([left[:n], right[:n][::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha * .30, lw=0,
                zorder=3)
    centre, _ = project(pts, K, cam_t, cam_R, h, w)
    if len(centre) > 1:
        ax.plot(centre[:, 0], centre[:, 1], ls, color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=4)
    return centre


def make_overlay(args, name, out):
    meta, ref, fixed, scaled, s = load_clip(args.clips, args.traj, name, args.repo)
    idx = NI.load_index(args.index)[name]
    raw = json.load(open(os.path.join(args.raw, "calibrated_sensor.json")))
    cs = next(c for c in raw if c["token"] == idx.cs_token)
    cam_t = np.asarray(cs["translation"], float)
    cam_R = NI.quat_to_rot(cs["rotation"])
    K = np.asarray(meta["K"], float)

    img = np.array(Image.open(os.path.join(args.clips, name, "images", "00000.jpg")))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=170)
    # dim a bright frame so the overlay reads; lift a dark one for the same reason
    lum = float(img.mean())
    base = img.astype(np.float32)
    base = base * .70 + 52 if lum > 110 else np.clip(base * 1.5 + 18, 0, 255)
    ax.imshow(base.astype(np.uint8))

    ribbon(ax, scaled, K, cam_t, cam_R, h, w, PR_C, .55, 1.7, ls="--")
    c_gt = ribbon(ax, ref, K, cam_t, cam_R, h, w, GT_C, .95, 3.2)
    c_pr = ribbon(ax, fixed, K, cam_t, cam_R, h, w, PR_C, .95, 2.8)

    # distance ticks along the true path
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    for metres in (10, 20, 30, 40, 50, 60):
        if metres > d[-1]:
            break
        i = int(np.searchsorted(d, metres))
        uv, ok = project(ref[i:i + 1], K, cam_t, cam_R, h, w)
        if len(uv) and 0 < uv[0, 0] < w - 40 and 0 < uv[0, 1] < h - 6:
            ax.plot(uv[0, 0], uv[0, 1], "o", color=GT_C, ms=4.5, zorder=6)
            ax.text(uv[0, 0] + 9, uv[0, 1], "%d m" % metres, color="white",
                    fontsize=7.5, va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc=GT_C, ec="none", alpha=.9))
    for c, col, lab in ((c_gt, GT_C, "true"), (c_pr, PR_C, "estimated")):
        if len(c) and 0 < c[-1, 0] < w and 0 < c[-1, 1] < h:
            ax.plot(c[-1, 0], c[-1, 1], "o", color=col, ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=1.2)

    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    handles = [plt.Line2D([], [], color=GT_C, lw=3.2, label="true ego path"),
               plt.Line2D([], [], color=PR_C, lw=2.8, label="estimated"),
               plt.Line2D([], [], color=PR_C, lw=1.7, ls="--", alpha=.6,
                          label="estimated, scale freed")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=.9,
              facecolor="white", edgecolor="none")
    ax.set_title("%s, both paths projected onto the first frame  "
                 "(%.0f m driven, ADE %.2f m)"
                 % (name, path_len(ref),
                    float(S.load_drivinggen(args.repo).align
                          .ade(fixed[None], ref[None]).ravel()[0])),
                 fontsize=10, color=INK, loc="left", pad=6)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/driving-gen/clips/dg")
    ap.add_argument("--traj", default="/root/driving-gen/traj/dg/predicted")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--raw", default="/root/driving-gen/nuscenes/raw/v1.0-trainval")
    ap.add_argument("--index", default="/root/driving-gen/nuscenes/index.json")
    ap.add_argument("--percl", default="out/dg_calibration/"
                                       "per_clip_dg_predicted_ego_fixed_scale.npz")
    ap.add_argument("--outdir", default="out/dg_calibration")
    ap.add_argument("--overlay", default="")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.percl, allow_pickle=True)
    ade, names = d["ade"], [str(x) for x in d["names"]]

    driven = []
    for n in names:
        with open(os.path.join(args.clips, n, "meta.json")) as f:
            driven.append(path_len(np.asarray(json.load(f)["ego_xy"], float)))
    driven = np.asarray(driven)
    moving = driven >= MOVING_M
    n_static = int((~moving).sum())
    print("moving clips: %d / %d | stationary (<%.0f m): %d"
          % (moving.sum(), len(names), MOVING_M, n_static))
    print("stationary clips, mean ADE %.3f" % ade[~moving].mean())

    m_idx = np.where(moving)[0]
    order = m_idx[np.argsort(ade[m_idx])]
    pick = [("best", 0), ("p25", int(.25 * len(order))), ("median", len(order) // 2),
            ("p75", int(.75 * len(order))), ("p95", int(.95 * len(order))),
            ("worst", len(order) - 1)]
    chosen = [(lab, names[order[i]]) for lab, i in pick]
    for lab, n in chosen:
        i = names.index(n)
        print("  %-7s %-14s ADE %6.2f  driven %5.1f m" % (lab, n, ade[i], driven[i]))

    make_bev(args, chosen, n_static, os.path.join(args.outdir, "traj_bev%s.png" % args.suffix))

    # The overlay is only legible on a well-lit frame, and a quarter of nuScenes
    # is shot at night, so pick the brightest clip whose error sits near the
    # median rather than the median clip itself.
    if args.overlay:
        pick_name = args.overlay
    else:
        med = float(np.median(ade[m_idx]))
        near = [i for i in m_idx if 0.8 * med <= ade[i] <= 1.25 * med]
        lums = []
        for i in near:
            p = os.path.join(args.clips, names[i], "images", "00000.jpg")
            lums.append(float(np.asarray(Image.open(p).convert("L")).mean()))
        pick_name = names[near[int(np.argmax(lums))]]
        print("overlay: %s (ADE %.2f, brightness %.0f of %d candidates near median)"
              % (pick_name, ade[names.index(pick_name)], max(lums), len(near)))
    make_overlay(args, pick_name, os.path.join(args.outdir, "traj_overlay%s.png" % args.suffix))


if __name__ == "__main__":
    main()
