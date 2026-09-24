#!/usr/bin/env python
"""V1 as trained against V2 as planned, in one figure.

Top row is `waypoint_vae_lr5x`: a clone of the stage-1 VAE encoder fine-tuned on
one clip-level waypoint target. Bottom row is the V2 of V2_PLAN.md: the same
grid, a new output conv, and four per-latent-frame heads, all supervised from
womd_7hz_f33 labels alone.

Usage:
    python conditioning/scripts/plot_v_arch.py --out out/v_arch.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import ACCENT, BLUE, FROZEN, GREY, INK, RED, arrow, canvas, frame, hbox

W, H = 21.0, 12.6
A, B = 10.05, 3.95          # row centres
H_BOX = 1.15


def box(ax, x, y, w, label, edge=BLUE, face="white", h=H_BOX, fs=9.0, weight="normal"):
    hbox(ax, x + w / 2, y, w, h, label, edge=edge, face=face, fs=fs, weight=weight)
    return x + w


def link(ax, x0, x1, y, color=BLUE, ls="-"):
    arrow(ax, (x0 + 0.06, y), (x1 - 0.06, y), color=color, ls=ls)


def note(ax, x, y, text, color=GREY, fs=8.3, ha="left", style="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            linespacing=1.45, style=style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/v_arch.png")
    args = ap.parse_args()
    fig, ax = canvas(plt, W, H)

    # ================================ V1 ================================
    frame(ax, 0.35, W - 0.35, A - 2.55, A + 2.15, edge=GREY, lw=1.1)
    ax.text(0.75, A + 1.75, "V1  waypoint_vae_lr5x, as trained", fontsize=12, color=INK, weight="bold")

    xs = 1.0
    x = box(ax, xs, A, 2.9, "16 raw frames\n96x96, 10 Hz")
    link(ax, x, x + 0.9, A)
    x = box(ax, x + 0.9, A, 3.5, "MiniWan encoder clone\ninit from vae_50e24b", face=FROZEN)
    link(ax, x, x + 0.9, A)
    x_r1 = x + 0.9
    x = box(ax, x_r1, A, 3.1, "r\n16 x 5 x 12 x 12", edge=ACCENT, face="#f0eaf7", weight="bold")
    link(ax, x, x + 0.9, A)
    x = box(ax, x + 0.9, A, 3.4, "query-decoder head\n128 wide, 2 layers", edge=RED)
    link(ax, x, x + 0.9, A)
    box(ax, x + 0.9, A, 3.6, "8 waypoints over 4 s\none prediction", edge=RED)

    note(ax, 1.0, A - 1.35,
         "One clip-level target: no latent frame has a meaning of its own.\n"
         "Trained on 16 context frames, used on 6 (k=2) and on 1 (k=1).\n"
         "KL 0.01 on r. Val waypoint ADE 1.513 m over 4 s.")
    note(ax, 12.2, A - 1.35,
         "Result at k=1: no separation from the null condition N\n"
         "on PSNR, ADE or DTW, and the worst FVD16 of the five.", color=RED)

    # ================================ V2 ================================
    frame(ax, 0.35, W - 0.35, B - 3.05, B + 2.15, edge=ACCENT, lw=1.3)
    ax.text(0.75, B + 1.75, "V2  planned", fontsize=12, color=INK, weight="bold")

    x = box(ax, xs, B, 2.9, "33 raw frames\n96x96, 7 Hz")
    link(ax, x, x + 0.8, B)
    x = box(ax, x + 0.8, B, 3.5, "MiniWan encoder\ntrunk from vae_miniwan_womd", face=FROZEN)
    link(ax, x, x + 0.8, B)
    x = box(ax, x + 0.8, B, 2.6, "new output conv\nrandom init")
    link(ax, x, x + 0.8, B)
    x_r2 = x + 0.8
    x_r2_end = box(ax, x_r2, B, 3.1, "r\n16 x 9 x 12 x 12", edge=ACCENT, face="#f0eaf7", weight="bold")

    # four training-only heads, stacked to the right of r
    heads = [("ego motion, every latent frame", 1.62),
             ("waypoints from r0..rk, k in 1..3\n6 hypotheses, closest one scored", 0.54),
             ("agent occupancy and velocity", -0.72),
             ("lane curvature, intersection, light", -1.62)]
    hx = x_r2_end + 1.15
    for label, dy in heads:
        h = 1.05 if "\n" in label else 0.72
        hbox(ax, hx + 2.6, B + dy, 5.2, h, label, edge=RED, face="white", fs=8.6)
        arrow(ax, (x_r2_end + 0.06, B + dy * 0.42), (hx - 0.06, B + dy), color=RED, rad=0.06)

    note(ax, 1.0, B - 2.0,
         "Causal: r_t sees only frames up to its own 4-frame chunk, so drawing k at random\n"
         "trains 1, 2 or 3 context latent frames with no masking. KL 1e-4 on r.\n"
         "Heads are training only and discarded; V outputs r alone. No distillation from D, T or G.")
    note(ax, 13.6, B - 2.25,
         "r keeps the grid of z, so it can be used either as\n"
         "cross-attention tokens or concatenated with z\n"
         "channel-wise for joint diffusion.", color=ACCENT)

    ax.text(0.75, 0.55, "Both encoders are causal and share z's grid: 8x spatial, 4x temporal, 16 channels. "
            "V2 changes what r is asked to represent, not its shape.", fontsize=9.2, color=INK)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print("wrote", out)


if __name__ == "__main__":
    main()
