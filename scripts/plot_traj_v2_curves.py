#!/usr/bin/env python
"""Traj-VAE v2 training curves from its history.json.

Left: the five task losses plus KL, on a log axis, since they span two orders of
magnitude. Right: validation waypoint ADE at k = 1, 2, 3 with the validation loss
behind it, and the epoch the best checkpoint came from.

    python scripts/plot_traj_v2_curves.py --run traj_v2 --out out/traj_v2_curves.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
COL = {"motion": "#2a78d6", "way": "#eb6834", "way_cls": "#f0a500", "occ": "#1baf7a",
       "curv": "#8e6bbf", "inter": "#e87ba4", "tl": "#7a8b99", "kl": "#b0b0a8"}
LAB = {"motion": "ego motion", "way": "waypoints, closest of 6", "way_cls": "hypothesis choice",
       "occ": "agent occupancy", "curv": "lane curvature", "inter": "intersection",
       "tl": "traffic light", "kl": "KL"}
KCOL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="traj_v2")
    ap.add_argument("--out", default="out/traj_v2_curves.png")
    args = ap.parse_args()
    h = json.loads((PROJ / "checkpoints/womd" / args.run / "history.json").read_text())
    ep = [r["epoch"] for r in h]
    best = min(h, key=lambda r: r["val_loss"])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.2))
    fig.patch.set_facecolor("white")

    style(axl)
    for key in COL:
        k = f"train_{key}"
        if k in h[0]:
            axl.plot(ep, [r[k] for r in h], color=COL[key], lw=1.9, label=LAB[key])
    axl.set_yscale("log")
    axl.set_xlabel("epoch", color=MUTED)
    axl.set_ylabel("training loss (unweighted)", color=MUTED)
    axl.set_title("Task losses, log scale", loc="left", fontsize=11, color=INK, pad=10)
    axl.legend(frameon=False, fontsize=8.5, ncol=2, labelcolor=INK)

    style(axr)
    for k in (1, 2, 3):
        axr.plot(ep, [r["val_ade"][str(k)] if isinstance(list(r["val_ade"])[0], str) else r["val_ade"][k]
                      for r in h], color=KCOL[k], lw=2.0, marker="o", ms=3.4,
                 label=f"val ADE, k={k} context frames")
    axr.axvline(best["epoch"], color=MUTED, lw=1.0, ls=":")
    axr.text(best["epoch"] + 0.3, axr.get_ylim()[1], f"best.pt, epoch {best['epoch']}",
             fontsize=8.5, color=MUTED, va="top")
    axr.set_xlabel("epoch", color=MUTED)
    axr.set_ylabel("waypoint ADE over 4 s (m)", color=MUTED)
    axr.set_title("Validation, 500 clips  (picked hypothesis)", loc="left", fontsize=11, color=INK, pad=10)
    ax2 = axr.twinx()
    ax2.plot(ep, [r["val_loss"] for r in h], color="#b0b0a8", lw=1.4, ls="--")
    ax2.set_ylabel("total val loss (dashed)", color=MUTED, fontsize=9)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    axr.legend(frameon=False, fontsize=8.5, labelcolor=INK)

    fig.suptitle(f"Traj-VAE v2 on womd_7hz_f33: {len(h)} epochs, encoder cloned from vae_miniwan_womd",
                 x=0.06, ha="left", fontsize=12.5, color=INK)
    out = PROJ / args.out
    fig.savefig(out, dpi=175, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    print("wrote", out)
    print(f"best epoch {best['epoch']}: val {best['val_loss']:.3f}, ADE " +
          " ".join(f"k{k} {best['val_ade'][str(k)] if isinstance(list(best['val_ade'])[0], str) else best['val_ade'][k]:.3f}"
                   for k in (1, 2, 3)))


if __name__ == "__main__":
    main()
