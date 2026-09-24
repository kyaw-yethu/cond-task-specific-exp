#!/usr/bin/env python
"""Plot the ego-motion probe's training curve from one or more run logs.

Reads the `[probe] ep N/M ...` lines a training run prints, so it works on a
captured stdout log without needing the run to have written anything else.

    python conditioning/scripts/plot_probe_curve.py clean=<log> vaeaug=<log> --out <png>
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"\[probe\] ep\s+(\d+)/(\d+) train_loss ([\d.]+)\s+"
    r"val ADE ([\d.]+) m\s+FDE ([\d.]+) m\s+heading ([\d.]+) deg\s+\((\d+)s\)")
FLOOR = re.compile(r"^\s+(real frames|VAE round-trip)\s+ADE\s+([\d.]+) m\s+FDE\s+([\d.]+) m")


def parse(path):
    eps, loss, ade, fde, hdg, secs, floors = [], [], [], [], [], [], {}
    for ln in Path(path).read_text(errors="ignore").splitlines():
        m = LINE.search(ln)
        if m:
            eps.append(int(m.group(1)))
            loss.append(float(m.group(3)))
            ade.append(float(m.group(4)))
            fde.append(float(m.group(5)))
            hdg.append(float(m.group(6)))
            secs.append(int(m.group(7)))
        f = FLOOR.match(ln)
        if f:
            floors[f.group(1)] = float(f.group(2))
    return dict(epoch=eps, loss=loss, ade=ade, fde=fde, heading=hdg, secs=secs, floors=floors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path/to/log")
    ap.add_argument("--out", default="probe_curve.png")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = parse(path)
        r = runs[label]
        print(f"{label}: {len(r['epoch'])} epochs, floors {r['floors']}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = {"clean": "#1f77b4", "vaeaug": "#d62728"}

    for label, r in runs.items():
        c = colours.get(label, None)
        if not r["epoch"]:
            continue
        ax[0].plot(r["epoch"], r["loss"], marker="o", ms=3, color=c, label=label)
        ax[1].plot(r["epoch"], r["ade"], marker="o", ms=3, color=c, label=f"{label} ADE")
        ax[1].plot(r["epoch"], r["fde"], marker="s", ms=3, ls="--", color=c, alpha=0.55,
                   label=f"{label} FDE")
        ax[2].plot(r["epoch"], r["heading"], marker="o", ms=3, color=c, label=label)
        for name, v in r["floors"].items():
            ax[1].axhline(v, ls=":", lw=1.2, color=c, alpha=0.8)
            ax[1].annotate(f"{label} {name}: {v:.3f}", (r["epoch"][-1], v),
                           textcoords="offset points", xytext=(-4, 4),
                           ha="right", fontsize=7, color=c)

    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("L1 (standardised + track)")
    ax[1].set_title("held-out displacement error")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metres"); ax[1].set_yscale("log")
    ax[2].set_title("held-out heading error")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("degrees"); ax[2].set_yscale("log")
    for a in ax:
        a.grid(alpha=0.25, which="both")
        a.legend(fontsize=7)

    fig.suptitle("Ego-motion probe: 40,000 Waymo clips, 2% tail of train held out; "
                 "dotted lines are test-split floors", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
