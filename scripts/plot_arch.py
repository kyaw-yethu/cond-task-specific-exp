#!/usr/bin/env python
"""The stage-1 architecture figure: one conditioned MiniDiT block.

Deliberately laid out like Phantom's figure so the two can be read side by side,
but with only the video branch, since stage 1 has no physics branch and no
coupling. The two substantive differences from that figure are marked: the
cross-attention context is a driving representation $r$ rather than a text
embedding, and there is no Vis-Attention, because nothing is coupled into this
branch.

Everything drawn here is read off the code, not the plan prose:
``f_toy.models.dit.Block`` for the sublayer order and ``cond_eval.cond_dit`` for
the conditioner. In particular the conditioner is a LayerNorm and a learned
positional embedding with no projection layer, because ``DIT_CROSSATTN_DIM`` is
set to the encoder's width and the block's own k/v projections do the mapping.

Usage:
    python scripts/plot_arch.py --out out/arch.png
"""
import argparse
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ACCENT, BLUE, FROZEN, GREY, INK, arrow, canvas, frame,
                      hbox, oplus, vbox)

BOX_W, BOX_H = 0.56, 2.15
CY = 4.10                 # vertical centre of the block chain
W, H = 20.2, 10.2         # canvas, in the same units every coordinate below uses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/arch.png")
    args = ap.parse_args()

    fig, ax = canvas(plt, W, H)

    # ================= the block chain =================
    chain = [
        ("box", "LayerNorm", BLUE, "white"),
        ("box", "Scale, Shift", BLUE, "white"),
        ("box", "Self-Attention", BLUE, "white"),
        ("box", "Gate", BLUE, "white"),
        ("plus",),
        ("box", "LayerNorm", BLUE, "white"),
        ("box", "Scale, Shift", BLUE, "white"),
        ("box", "Cross-Attention", ACCENT, "#f0eaf7"),
        ("box", "Gate", BLUE, "white"),
        ("plus",),
        ("box", "LayerNorm", BLUE, "white"),
        ("box", "Scale, Shift", BLUE, "white"),
        ("box", "FFN", BLUE, "white"),
        ("box", "Gate", BLUE, "white"),
        ("plus",),
    ]

    x = 5.95
    gap = 0.17
    centres, plus_x, scale_x, gate_x = {}, [], [], []
    prev_right = None
    for item in chain:
        if item[0] == "box":
            _, label, edge, face = item
            cx = x + BOX_W / 2
            if prev_right is not None:
                arrow(ax, (prev_right, CY), (cx - BOX_W / 2, CY))
            vbox(ax, cx, CY, label, edge=edge, face=face,
                 w=BOX_W, h=BOX_H)
            centres.setdefault(label, []).append(cx)
            if label == "Scale, Shift":
                scale_x.append(cx)
            if label == "Gate":
                gate_x.append(cx)
            prev_right = cx + BOX_W / 2
            x = prev_right + gap
        else:
            cx = x + 0.115
            arrow(ax, (prev_right, CY), (cx - 0.115, CY))
            oplus(ax, cx, CY)
            plus_x.append(cx)
            prev_right = cx + 0.115
            x = prev_right + gap

    # ---- the residual paths, dashed, as the reference draws them ----
    entry = 5.95 - gap
    top, bot = CY + BOX_H / 2 + 0.30, CY - BOX_H / 2 - 0.30
    for src, dst, lane in ((entry, plus_x[0], top),
                           (plus_x[0], plus_x[1], bot),
                           (plus_x[1], plus_x[2], top)):
        ax.plot([src, src], [CY, lane], ls="--", lw=1.1, color=BLUE, zorder=1)
        ax.plot([src, dst], [lane, lane], ls="--", lw=1.1, color=BLUE, zorder=1)
        arrow(ax, (dst, lane), (dst, CY + (0.115 if lane > CY else -0.115)),
              ls="--", lw=1.1, z=1)

    # ---- the repeated-block frame ----
    x0, x1 = 5.45, prev_right + 0.32
    frame(ax, x0, x1, CY - BOX_H / 2 - 0.78, CY + BOX_H / 2 + 0.78)
    ax.text(x1 - 0.30, CY - BOX_H / 2 - 1.02, r"$\times\ 6$", fontsize=13,
            ha="right", va="center", color=INK)

    # ================= left: the video path =================
    hbox(ax, 1.05, 6.35, 1.75, 0.92,
         "Input Video\n16 x 96 x 96", edge=GREY)
    arrow(ax, (1.05, 5.89), (1.05, 5.16), color=GREY)

    ax.add_patch(Polygon([(0.28, 4.94), (1.82, 4.94), (1.55, 4.02), (0.55, 4.02)],
                         closed=True, linewidth=1.3, edgecolor=BLUE,
                         facecolor=FROZEN, zorder=3))
    ax.text(1.05, 4.48, "MiniWan VAE\nEncoder", ha="center", va="center",
            fontsize=8.5, color=INK, zorder=4, linespacing=1.3)
    ax.text(1.05, 3.79, "frozen", ha="center", va="center", fontsize=7.5,
            color=GREY, style="italic")

    ax.text(2.66, 4.76, r"clean latents $x_0$", fontsize=8.5, color=INK,
            ha="center")
    ax.text(2.66, 4.49, r"$5\times12\times12$", fontsize=8, color=GREY,
            ha="center")
    arrow(ax, (1.82, 4.30), (2.12, 4.30), color=BLUE)

    # noise, and the Video2World splice
    hbox(ax, 1.15, 1.88, 2.10, 0.70, "Flow-matching Noise", edge=GREY, fs=8.5)
    arrow(ax, (2.20, 1.88), (3.66, 1.88), color=BLUE)
    arrow(ax, (3.66, 1.88), (3.66, CY - 0.14), color=BLUE)

    arrow(ax, (3.20, 4.30), (3.66, 4.30), color=BLUE)
    arrow(ax, (3.66, 4.30), (3.66, CY + 0.14), color=BLUE)
    oplus(ax, 3.66, CY, sym=r"$\odot$", r=0.135)
    ax.text(3.49, CY - 0.44, "splice", fontsize=7.8, color=GREY, ha="right",
            va="top")

    hbox(ax, 4.85, 1.88, 1.30, 0.70, "Condition\nMask", edge=GREY, fs=8)
    arrow(ax, (4.85, 2.23), (4.85, CY - BOX_H / 2 - 0.02), color=GREY)

    vbox(ax, 4.85, CY, "Patchify", edge=BLUE, w=0.60, h=BOX_H)
    arrow(ax, (3.80, CY), (4.55, CY), color=BLUE)
    arrow(ax, (5.15, CY), (entry + 0.17, CY), color=BLUE)
    ax.text(4.85, CY + BOX_H / 2 + 0.30, "720 tokens\nx 256", fontsize=7.5,
            color=GREY, ha="center", va="bottom", linespacing=1.3)

    # ================= top: the r path =================
    cross_x = centres["Cross-Attention"][0]
    ry = 7.85
    ex = cross_x - 4.30                       # the encoder column

    hbox(ax, ex, ry + 0.80, 2.70, 0.74,
         r"Context Frames only:  raw $0..4$", edge=GREY, fs=8.5)
    arrow(ax, (ex, ry + 0.43), (ex, ry + 0.02), color=GREY)
    ax.text(ex + 0.16, ry + 0.24, "no future leakage", fontsize=7.5,
            color=GREY, ha="left", style="italic")

    hbox(ax, ex, ry - 0.38, 3.00, 0.78,
         r"Frozen $r$ Encoder", edge=ACCENT, face=FROZEN, fs=9.5,
         weight="bold")
    arrow(ax, (ex + 1.50, ry - 0.38), (cross_x - 1.42, ry - 0.38),
          color=ACCENT)
    ax.text((ex + 1.50 + cross_x - 1.42) / 2, ry - 0.24,
            r"$r$ tokens, $N \times d$", fontsize=8, color=GREY,
            ha="center", va="bottom")

    hbox(ax, cross_x, ry - 0.38, 2.80, 0.78,
         "LayerNorm  +\nLearned Positional Emb", edge=ACCENT, fs=8.5)

    arrow(ax, (cross_x, ry - 0.77), (cross_x, CY + BOX_H / 2 + 0.02),
          color=ACCENT)
    ax.text(cross_x + 0.16, CY + BOX_H / 2 + 0.40, "k, v", fontsize=8.5,
            color=ACCENT, ha="left")

    # ================= bottom: AdaLN-zero =================
    ay, ax_c = 1.34, (x0 + x1) / 2
    ax.text(ax_c, 0.66, r"$t$", fontsize=12, ha="center", color=INK)
    arrow(ax, (ax_c, 0.84), (ax_c, ay - 0.31), color=GREY)
    hbox(ax, ax_c, ay, 2.05, 0.62, "AdaLN-Zero", edge=GREY, fs=9.5)

    for tx in scale_x + gate_x:
        arrow(ax, (ax_c, ay + 0.31), (tx, CY - BOX_H / 2 - 0.02), color=GREY,
              lw=1.0, rad=0.13 if tx < ax_c else -0.13)

    # ================= right: out =================
    ox = prev_right + 0.32
    arrow(ax, (ox, CY), (ox + 0.52, CY), color=BLUE)
    hbox(ax, ox + 1.32, CY, 1.50, 1.10, "Final Layer\nUnpatchify", edge=BLUE,
         fs=8.5)
    arrow(ax, (ox + 2.07, CY), (ox + 2.55, CY), color=BLUE)
    ax.text(ox + 2.66, CY, "Video\nVelocity", fontsize=9.5, ha="left",
            va="center", color=INK, linespacing=1.35)

    # ================= the five conditions =================
    ly, lx = 9.42, 0.28
    ax.text(lx, ly + 0.46,
            r"The five conditions differ only in the frozen $r$ encoder",
            fontsize=10, color=INK, ha="left", va="top", weight="bold")
    rows = [
        ("N", "null context", r"a learned single token", "1"),
        ("D", "Drive-JEPA domain", r"ViT-L, $d = 1024$", "768"),
        ("T", "Drive-JEPA task", r"ViT-L, $d = 1024$", "768"),
        ("V", "Traj-VAE", r"waypoint VAE, $d = 16$", "288"),
        ("G", "V-JEPA2 (generic)", r"ViT-H, $d = 1280$", "768"),
    ]
    for i, (tag, name, detail, ntok) in enumerate(rows):
        y = ly - 0.34 - i * 0.30
        ax.text(lx + 0.06, y, tag, fontsize=9, color=ACCENT, weight="bold",
                ha="left", va="top")
        ax.text(lx + 0.40, y, name, fontsize=8.5, color=INK, ha="left", va="top")
        ax.text(lx + 2.05, y, detail, fontsize=8.5, color=GREY, ha="left",
                va="top")
        ax.text(lx + 4.10, y, ntok, fontsize=8.5, color=GREY, ha="right",
                va="top")
    ax.text(lx + 4.10, ly - 0.04, r"tokens at $k{=}2$", fontsize=7.5,
            color=GREY, ha="right", va="top", style="italic")

    ax.text(lx, 0.22,
            r"No physics branch and no Vis-Attention: stage 1 denoises video only, so $r$ "
            r"enters through cross-attention alone, in place of the text embedding a video "
            r"DiT would normally take." + "\n" +
            r"The splice keeps the first $k$ latent frames clean, $k \in \{1, 2\}$, and $r$ "
            r"sees exactly the raw frames those cover, so no future information reaches the "
            r"cross-attention.",
            fontsize=8.5, color=GREY, ha="left", va="center", linespacing=1.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white", bbox_inches="tight",
                pad_inches=0.18)
    print("wrote", out)


if __name__ == "__main__":
    main()
