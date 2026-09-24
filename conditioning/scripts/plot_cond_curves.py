#!/usr/bin/env python
"""Training curves for the five stage-1 conditions, from their history.json.

Two panels rather than one, because a single linear axis cannot show both things
worth seeing. The left panel is the whole run on a log axis, where the only
visible feature is that every condition makes the same early descent. The right
panel is the last sixty epochs at the scale the differences actually live at,
about 0.02 of flow loss, which is invisible on the left.

``val_flow`` exists only every fifth epoch (``--val-every 5``); the other epochs
carry NaN and are dropped rather than interpolated.

    python conditioning/scripts/plot_cond_curves.py --ckpt-dir checkpoints/waymo --out out/cond_curves.png
    python conditioning/scripts/plot_cond_curves.py --suffix 1 --out out/cond_curves_k1.png \
        --title "Flow-matching loss, k=1 on dit_large, one seed"

A run still training is drawn up to its last finished epoch.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical slots 1-5, in fixed order, unchanged. Documented as
# passing every hard gate on the adjacent pairlist in light mode, which is this
# case: lines on a light surface.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ORDER = ["N", "D", "T", "V", "G"]
NAME = {"N": "N  null context",
        "D": "D  Drive-JEPA domain",
        "T": "T  Drive-JEPA task",
        "V": "V  Traj-VAE",
        "G": "G  V-JEPA2 ViT-H"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
ZOOM_FROM = 40


def load(ckpt_dir, tag):
    h = json.loads((Path(ckpt_dir) / tag / "history.json").read_text())
    ep = [e["epoch"] for e in h]
    tr = [e["train_flow"] for e in h]
    va = [(e["epoch"], e["val_flow"]) for e in h
          if e["val_flow"] == e["val_flow"]]          # drops the NaN epochs
    return ep, tr, va


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/waymo")
    ap.add_argument("--out", default="out/cond_curves.png")
    ap.add_argument("--suffix", default="",
                    help="tag suffix of a variant's runs, e.g. 1 for N1 D1 T1 V1 G1")
    ap.add_argument("--title", default="Flow-matching loss, five stage-1 conditions, one seed")
    args = ap.parse_args()

    runs = {}
    for tag in ORDER:
        try:
            runs[tag] = load(args.ckpt_dir, tag + args.suffix)
        except FileNotFoundError:
            print(f"skip {tag}{args.suffix}: no history.json")
    if not runs:
        raise SystemExit(f"no history.json under {args.ckpt_dir}")
    tags = [t for t in ORDER if t in runs]
    colour = {t: PALETTE[i] for i, t in enumerate(ORDER)}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.4, 5.5),
                                   gridspec_kw=dict(width_ratios=[1, 1.18],
                                                    wspace=0.22))
    fig.patch.set_facecolor("white")

    # ---------------- left: the whole run ----------------
    style(axl)
    for t in tags:
        ep, tr, va = runs[t]
        axl.plot(ep, tr, color=colour[t], linewidth=2.0, label=NAME[t].replace(t, t + args.suffix, 1))
        axl.plot([e for e, _ in va], [v for _, v in va], color=colour[t],
                 linewidth=0, marker="o", markersize=3.4, alpha=0.55)
    axl.set_yscale("log")
    axl.set_xlabel("epoch", fontsize=10, color=MUTED)
    axl.set_ylabel("flow-matching loss", fontsize=10, color=MUTED)
    axl.set_title("All 100 epochs, log scale",
                  fontsize=11, color=INK, loc="left", pad=10)
    axl.set_xlim(0, 101)
    axl.axvspan(ZOOM_FROM, 101, color="#f0f0ec", zorder=0)
    axl.text(ZOOM_FROM + 2, axl.get_ylim()[1] * 0.93, "shown right",
             fontsize=8.5, color=MUTED, va="top")
    axl.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right",
               handlelength=1.6)
    axl.set_xlabel("epoch\nlines: train_flow      dots: val_flow, every 5th epoch",
                   fontsize=10, color=MUTED, linespacing=1.9)

    # ---------------- right: where the separation lives ----------------
    style(axr)
    lo, hi = 1.0, 0.0
    ends = []
    for t in tags:
        ep, tr, va = runs[t]
        tx = [e for e in ep if e >= ZOOM_FROM]
        ty = [v for e, v in zip(ep, tr) if e >= ZOOM_FROM]
        axr.plot(tx, ty, color=colour[t], linewidth=2.0, alpha=0.38)
        vx = [e for e, _ in va if e >= ZOOM_FROM]
        vy = [v for e, v in va if e >= ZOOM_FROM]
        axr.plot(vx, vy, color=colour[t], linewidth=2.0, marker="o",
                 markersize=4.6, markeredgecolor="white", markeredgewidth=0.9)
        lo, hi = min(lo, min(ty + vy)), max(hi, max(ty + vy))
        if vy:
            best = min(v for _, v in va)
            ends.append((vy[-1], t, best))

    pad = (hi - lo) * 0.10
    axr.set_ylim(lo - pad, hi + pad)
    axr.set_xlim(ZOOM_FROM, 122)
    axr.set_xticks([40, 50, 60, 70, 80, 90, 100])
    axr.set_xlabel("epoch", fontsize=10, color=MUTED)
    axr.set_title(f"Epochs {ZOOM_FROM} to 100, linear   "
                  "(bold: val_flow, faded: train_flow)",
                  fontsize=11, color=INK, loc="left", pad=10)

    # the best checkpoint, which every condition happened to reach together
    best_eps = {min(va, key=lambda p: p[1])[0] for _, _, va in runs.values()}
    if len(best_eps) == 1:
        be = best_eps.pop()
        axr.axvline(be, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
        axr.text(be - 1.0, axr.get_ylim()[1], f"best.pt, epoch {be}\nfor all {len(runs)}",
                 fontsize=8.5, color=MUTED, ha="right", va="top",
                 linespacing=1.4)

    # end labels, pushed apart so five near-identical values stay legible
    span = axr.get_ylim()[1] - axr.get_ylim()[0]
    gap = span * 0.085
    ends.sort(key=lambda r: -r[0])
    placed = []
    for y0, t, best in ends:
        y = placed[-1] - gap if placed and placed[-1] - y0 < gap else y0
        placed.append(y)
        axr.annotate(f"{t}{args.suffix}   {best:.4f}", (100, y0), xytext=(105.5, y),
                     fontsize=9, color=INK, va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color=colour[t],
                                     linewidth=1.1, shrinkA=2, shrinkB=3))
    axr.text(104.5, axr.get_ylim()[1] - span * 0.03, "best val_flow",
             fontsize=8.5, color=MUTED, ha="left", va="top", style="italic")

    fig.suptitle(args.title,
                 fontsize=13, color=INK, x=0.072, ha="left", y=0.975)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white", bbox_inches="tight",
                pad_inches=0.22)
    print("wrote", out)

    print("\nbest val_flow")
    for t in tags:
        _, _, va = runs[t]
        print(f"  {t}  {min(v for _, v in va):.6f}  at epoch "
              f"{min(va, key=lambda p: p[1])[0]}")


if __name__ == "__main__":
    main()
