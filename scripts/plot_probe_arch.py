#!/usr/bin/env python
"""The ego-motion probe's architecture, drawn in the same language as arch.png.

Read off ``cond_eval/pose_probe.py``, so the channel widths, kernel sizes and
strides are the ones that actually run. The figure is built around the single
design decision that matters: the network predicts a per-pair increment and the
geometry composes it, rather than the network regressing 16 absolute poses. That
is why the chain splits into a learned half and a fixed-geometry half, drawn in
different colours.

    python scripts/plot_probe_arch.py --out out/probe_arch.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ACCENT, BLUE, GREY, INK, RED, arrow, canvas, frame,
                      hbox, vbox)

BOX_W, BOX_H = 0.60, 2.05
CY = 4.35
W, H = 19.0, 8.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/probe_arch.png")
    args = ap.parse_args()

    fig, ax = canvas(plt, W, H)

    # ---------------- input ----------------
    hbox(ax, 1.20, CY, 1.95, 1.15, "Clip\n16 x 3 x 96 x 96", edge=GREY)
    arrow(ax, (2.18, CY), (2.76, CY), color=BLUE)

    x = 3.06
    gap = 0.20

    def step(label, sub=None, edge=BLUE, face="white", w=BOX_W):
        nonlocal x
        cx = x + w / 2
        vbox(ax, cx, CY, label, edge=edge, face=face, w=w, h=BOX_H, sub=sub)
        x = cx + w / 2 + gap
        return cx

    def link(a, b, color=BLUE):
        arrow(ax, (a + BOX_W / 2, CY), (b - BOX_W / 2, CY), color=color)

    pair_x = step("Pair Stacking", "adjacent frames -> 6 ch")
    ax.text(pair_x, CY + BOX_H / 2 + 0.22, "15 x (6, 96, 96)", fontsize=8,
            color=GREY, ha="center", va="bottom")

    # ---------------- the weight-shared 2D trunk ----------------
    x += 0.24
    trunk_x0 = x - 0.16
    blocks = [("Conv 7x7  /2", "6 -> 32"),
              ("Conv 5x5  /2", "32 -> 64"),
              ("Conv 3x3  /2", "64 -> 128"),
              ("Conv 3x3  /2", "128 -> 256"),
              ("Conv 3x3  /2", "256 -> 256")]
    prev = pair_x
    trunk_centres = []
    for label, sub in blocks:
        cx = step(label, sub)
        link(prev, cx)
        prev = cx
        trunk_centres.append(cx)
    trunk_x1 = x - gap + 0.16

    frame(ax, trunk_x0, trunk_x1, CY - BOX_H / 2 - 0.62, CY + BOX_H / 2 + 0.62)
    ax.text((trunk_x0 + trunk_x1) / 2, CY + BOX_H / 2 + 0.76,
            "shared 2D trunk, one forward for all 15 pairs",
            fontsize=9, color=ACCENT, ha="center", va="bottom", weight="bold")
    ax.text((trunk_x0 + trunk_x1) / 2, CY - BOX_H / 2 - 0.76,
            "every block: Conv2d -> GroupNorm -> SiLU        "
            "96 -> 48 -> 24 -> 12 -> 6 -> 3 px",
            fontsize=8.5, color=GREY, ha="center", va="top")

    # ---------------- pooling and the temporal stage ----------------
    x += 0.24
    mean_x = step("Spatial Mean", "over 3 x 3")
    link(prev, mean_x)
    ax.text(mean_x, CY + BOX_H / 2 + 0.22, "(15, 256)", fontsize=8,
            color=GREY, ha="center", va="bottom")

    x += 0.22
    temp_x0 = x - 0.16
    t1 = step("Conv1d  k3", "over t")
    link(mean_x, t1)
    t2 = step("Conv1d  k3", "over t")
    link(t1, t2)
    temp_x1 = x - gap + 0.16
    frame(ax, temp_x0, temp_x1, CY - BOX_H / 2 - 0.34, CY + BOX_H / 2 + 0.34,
          edge=GREY, lw=1.1)
    ax.text((temp_x0 + temp_x1) / 2, CY + BOX_H / 2 + 0.48,
            "neighbouring pairs inform each other", fontsize=8, color=GREY,
            ha="center", va="bottom")

    x += 0.20
    head_x = step("Head  1x1", "zero-init", edge=ACCENT, face="#f0eaf7")
    link(t2, head_x)
    ax.text(head_x, CY + BOX_H / 2 + 0.22, "(15, 3)", fontsize=8, color=GREY,
            ha="center", va="bottom")

    # ---------------- the fixed-geometry half ----------------
    x += 0.30
    den_x = step("Denormalise", r"$\hat{\iota}\cdot\sigma + \mu$", edge=RED)
    link(head_x, den_x, color=RED)
    se_x = step("Compose SE(2)", "15 increments -> 16 poses", edge=RED)
    link(den_x, se_x, color=RED)

    geo_x0, geo_x1 = den_x - BOX_W / 2 - 0.16, se_x + BOX_W / 2 + 0.16
    frame(ax, geo_x0, geo_x1, CY - BOX_H / 2 - 0.62, CY + BOX_H / 2 + 0.62,
          edge=RED, lw=1.2)
    ax.text((geo_x0 + geo_x1) / 2, CY + BOX_H / 2 + 0.76,
            "fixed geometry, no learned weights",
            fontsize=9, color=RED, ha="center", va="bottom", weight="bold")

    arrow(ax, (se_x + BOX_W / 2, CY), (se_x + BOX_W / 2 + 0.66, CY), color=RED)
    ax.text(se_x + BOX_W / 2 + 0.80, CY,
            "16 poses,\nframe-0 body frame", fontsize=9.5, color=INK,
            ha="left", va="center", linespacing=1.4)

    # ---------------- what the head predicts ----------------
    # ---------------- the objective ----------------
    ly = 1.42
    lcx = (mean_x + se_x) / 2
    hbox(ax, lcx, ly, 7.60, 0.74,
         r"loss:   L1 on standardised increments  $+\ 0.2 \times$ L1 on the "
         r"integrated track", edge=GREY, fs=9)
    arrow(ax, (head_x, CY - BOX_H / 2 - 0.02), (head_x, ly + 0.39),
          color=GREY, lw=1.0, rad=-0.10)
    arrow(ax, (se_x, CY - BOX_H / 2 - 0.02), (se_x, ly + 0.39),
          color=GREY, lw=1.0, rad=0.10)
    ax.text(lcx, ly - 0.52,
            "integration error is penalised directly, not only per step",
            fontsize=8.5, color=GREY, ha="center", va="top")

    # ---------------- the splice, which is why increments are predicted ----
    ax.annotate("at evaluation, integration starts from the TRUE pose\n"
                "at the conditioning cutoff, so every measured increment\n"
                "comes from generated frames alone",
                xy=(se_x, CY + BOX_H / 2 + 0.04),
                xytext=(se_x - 0.10, CY + BOX_H / 2 + 1.45),
                fontsize=8.5, color=RED, ha="center", va="bottom",
                linespacing=1.45,
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.1,
                                shrinkA=4, shrinkB=2))

    # ---------------- notes ----------------
    ax.text(0.28, H - 0.30,
            "Ego-motion probe: 1,415,043 parameters",
            fontsize=11, color=INK, ha="left", va="top", weight="bold")
    ax.text(0.28, H - 0.72,
            "Trained from scratch on raw frames, sharing no weights with the VAE\n"
            "or any candidate encoder, so nothing in the comparison is circular.\n"
            "Zero-init head: a fresh probe predicts the training mean, not noise.",
            fontsize=8.5, color=GREY, ha="left", va="top", linespacing=1.6)

    ax.text(0.28, 0.34,
            "One trunk is applied to all 15 pairs because the per-pair problem is the "
            "same at every $t$: the 15 steps share supervision instead of one clip "
            "giving one example, and the network\nnever has to learn to integrate. The "
            r"head predicts $(\Delta \mathrm{fwd},\ \Delta \mathrm{lat},\ \Delta "
            r"\theta)$ in the EARLIER frame's body frame, which is what lets "
            "integration be restarted from the true pose at the cutoff.\n"
            "Valid only for this render configuration, since the fixed camera over a "
            "flat road is what makes metric scale observable.",
            fontsize=8.5, color=GREY, ha="left", va="center", linespacing=1.7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white", bbox_inches="tight",
                pad_inches=0.18)
    print("wrote", out)


if __name__ == "__main__":
    main()
