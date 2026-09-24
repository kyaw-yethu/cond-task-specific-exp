#!/usr/bin/env python
"""Figure: the probe's architecture, its training objective, and the splice.

Panel A is schematic. Panels B and C are drawn from the trained probe on real
test clips, so the objective is shown on actual predictions rather than a
cartoon.

Usage:
    python scripts/plot_probe_design.py --jdm third_party/Task_specific_JDM \\
        --probe <proj>/out/probe_vaeaug/probe.pt \\
        --dataset-dir <proj>/data/driving_wp_f16/test --out <png>
"""
import argparse
import sys
from pathlib import Path


def _bootstrap(jdm):
    if not Path(jdm, "f_toy").is_dir():
        sys.exit(f"no f_toy under {jdm}")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BLUE, RED, GREY, GREEN = "#1f77b4", "#d62728", "#888888", "#2ca02c"


def panel_architecture(ax):
    import matplotlib.patches as mp

    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("A. architecture: one weight-shared trunk over 15 frame pairs",
                 fontsize=10, loc="left")

    def box(x, y, w, h, label, sub="", fc="white", ec=BLUE):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                       fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), label,
                ha="center", va="center", fontsize=8.5)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center",
                    fontsize=7, color=GREY)

    def arrow(x0, y0, x1, y1, c=GREY):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))

    # input clip
    for i in range(5):
        ax.add_patch(mp.Rectangle((0.15 + i * 0.16, 4.4), 0.5, 0.75,
                                  fc="#dce6f2", ec=BLUE, lw=0.7))
    ax.text(0.75, 3.98, "clip\n16 x 3 x 96 x 96", ha="center", va="top",
            fontsize=7.5, color=GREY)

    box(1.7, 4.35, 1.5, 0.9, "adjacent pairs", "15 x (6, 96, 96)", fc="#dce6f2")
    arrow(1.3, 4.8, 1.68, 4.8)

    box(3.5, 4.35, 2.5, 0.9, "shared 2D CNN trunk",
        "96>48>24>12>6>3 px, GN + SiLU", fc="#eef3fa")
    ax.text(4.75, 5.42, "one trunk applied to all 15 pairs: the per-pair problem is "
                        "the same at every t", ha="center", va="bottom", fontsize=7,
            color=GREY)
    arrow(3.22, 4.8, 3.48, 4.8)

    box(6.3, 4.35, 1.3, 0.9, "spatial mean", "(15, 256)")
    arrow(6.02, 4.8, 6.28, 4.8)

    box(7.9, 4.35, 1.9, 0.9, "2 x Conv1d over t",
        "neighbouring pairs inform\neach other", fc="#eef3fa")
    arrow(7.62, 4.8, 7.88, 4.8)

    box(7.9, 2.9, 1.9, 0.8, "head, zero-init", "-> (15, 3)", ec=RED)
    arrow(8.85, 4.3, 8.85, 3.72, RED)
    ax.text(8.85, 2.66, "zero-init: a fresh probe predicts\nthe training mean, "
                        "not noise", ha="center", va="top", fontsize=7, color=GREY)

    box(5.0, 2.9, 2.5, 0.8, r"denormalise: $\hat{\iota}\cdot\sigma + \mu$",
        "metres, radians", ec=RED)
    arrow(7.88, 3.3, 7.52, 3.3, RED)

    box(2.0, 2.9, 2.6, 0.8, "compose as SE(2)",
        "15 increments -> 16 poses", ec=RED)
    arrow(4.98, 3.3, 4.62, 3.3, RED)

    arrow(1.98, 3.3, 1.66, 3.3, RED)
    ax.text(1.56, 3.3, "16 poses,\nframe-0 body frame",
            fontsize=7.5, va="center", ha="right", color=RED)
    ax.text(6.25, 2.66, r"increments $(\Delta\mathrm{fwd}, \Delta\mathrm{lat}, "
                        r"\Delta\theta)$ in the EARLIER frame's body frame",
            fontsize=7.5, va="top", ha="center", color=RED)

    ax.text(0.15, 1.95, "1,415,043 parameters. Trained from scratch on raw frames, "
                        "sharing no weights with the VAE or any candidate encoder,\n"
                        "so nothing in the comparison is circular.",
            fontsize=7.5, color="black")
    ax.text(0.15, 1.15, "Per-step rather than 16 absolute positions: the 15 steps share "
                       "supervision instead of one clip giving one example, the network\n"
                       "never has to learn to integrate, and integration can begin "
                       "anywhere (panel C).", fontsize=7.5, color=GREY)


def panel_objective(ax, tgt_inc, pred_inc, tgt_xy, pred_xy):
    import numpy as np

    ax.set_title("B. objective: two L1 terms, one per representation of the same motion",
                 fontsize=10, loc="left")
    ax.plot(tgt_xy[:, 1], tgt_xy[:, 0], "-o", ms=3.5, color=GREY, lw=2,
            label="true track (from labels)")
    ax.plot(pred_xy[:, 1], pred_xy[:, 0], "-o", ms=3, color=RED, lw=1.4,
            label="predicted track (composed)")

    # per-step increments, drawn as the vectors they are
    for t in range(0, len(tgt_inc), 2):
        ax.annotate("", (tgt_xy[t + 1, 1], tgt_xy[t + 1, 0]),
                    (tgt_xy[t, 1], tgt_xy[t, 0]),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1, alpha=0.8))
    ax.plot([], [], color=BLUE, lw=1.1, label=r"per-step increment $\iota_t$")

    ade = float(np.linalg.norm(pred_xy - tgt_xy, axis=-1).mean())
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("forward (m)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.03,
            "$\\mathcal{L} = \\mathrm{L1}(\\hat{\\iota},\\ \\iota)_{\\mathrm{std}}"
            "\\ +\\ 0.2\\,\\mathrm{L1}(\\hat{p},\\ p)\\,/\\,10\\,\\mathrm{m}$\n"
            "term 1 on the increments (blue arrows)\n"
            "term 2 on the composed track (red vs grey)\n"
            f"this clip: ADE {ade:.3f} m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(fc="white", ec=GREY, alpha=0.9, boxstyle="round,pad=0.4"))


def panel_splice(ax, tgt_xy, pred_xy, spliced_xy, cutoff):
    ax.set_title(f"C. at evaluation: integrate from the true pose at the cutoff "
                 f"(k=2, raw frame {cutoff})", fontsize=10, loc="left")
    ax.plot(tgt_xy[:, 1], tgt_xy[:, 0], "-o", ms=3.5, color=GREY, lw=2,
            label="true track")
    ax.plot(pred_xy[:, 1], pred_xy[:, 0], "--o", ms=2.5, color=RED, lw=1.1,
            alpha=0.55, label="from the origin (drifts)")
    ax.plot(spliced_xy[:, 1], spliced_xy[:, 0], "-o", ms=3, color=GREEN, lw=1.6,
            label="spliced at the cutoff")
    ax.plot(tgt_xy[:cutoff, 1], tgt_xy[:cutoff, 0], "o", ms=7, mfc="none",
            mec=GREEN, mew=1.4, label="context frames (ground truth)")
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("forward (m)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.03,
            "Only increments over GENERATED frames\n"
            "contribute, so no error accumulates out\n"
            "of the real context. This is what the\n"
            "per-step parameterisation buys.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(fc="white", ec=GREEN, alpha=0.9, boxstyle="round,pad=0.4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jdm", default="third_party/Task_specific_JDM")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default="probe_design.png")
    ap.add_argument("--clip", type=int, default=None,
                    help="default: the clip with the largest heading change, so the "
                         "rotation terms are visible")
    args = ap.parse_args()

    _bootstrap(args.jdm)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from f_toy.data import get_dataset_plugin
    from cond_eval.pose_probe import load_pose_probe
    from cond_eval.traj_metrics import increments_from_labels, integrate_increments

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_pose_probe(args.probe, dev)

    ds = get_dataset_plugin({"DATASET": "waymo"}).DiskDataset(args.dataset_dir)
    ann = np.load(Path(args.dataset_dir) / "annotations.npz")
    pos, hdg = ann["ego_position"], ann["ego_heading"]
    turn = np.abs((hdg[:, -1] - hdg[:, 0] + np.pi) % (2 * np.pi) - np.pi)
    i = args.clip if args.clip is not None else int(np.argsort(-turn)[3])
    print(f"clip {i}: turn {np.degrees(turn[i]):.1f} deg")

    video = ds[i][0][None].to(dev)
    tgt_inc = increments_from_labels(pos[i:i + 1], hdg[i:i + 1])
    pred_inc = model.predict_increments(video)
    tgt_xy, tgt_h = integrate_increments(tgt_inc)
    pred_xy, _ = integrate_increments(pred_inc)

    cutoff = 5                      # k=2 with 4x temporal compression
    tail_xy, _ = integrate_increments(pred_inc[:, cutoff - 1:],
                                      tgt_xy[:, cutoff - 1], tgt_h[:, cutoff - 1])
    spliced = np.concatenate([tgt_xy[:, :cutoff - 1], tail_xy], axis=1)

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.22)
    panel_architecture(fig.add_subplot(gs[0, :]))
    panel_objective(fig.add_subplot(gs[1, 0]), tgt_inc[0], pred_inc[0],
                    tgt_xy[0], pred_xy[0])
    panel_splice(fig.add_subplot(gs[1, 1]), tgt_xy[0], pred_xy[0], spliced[0], cutoff)
    fig.suptitle("Ego-motion probe: architecture, objective, and evaluation-time "
                 f"integration  (test clip {i})", fontsize=11)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
