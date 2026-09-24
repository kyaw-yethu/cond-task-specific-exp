#!/usr/bin/env python
"""Per-frame error of one trained condition, on train clips and test clips.

Answers two questions about a checkpoint without retraining anything:

1. Does it generate as well on clips it trained on as on held-out clips? A large
   train/test gap means generalisation (data) limits quality; poor samples on
   both mean the model itself (capacity, optimisation) does.
2. How does error grow with frame index? Poor frames right after the context
   point at the backbone or the VAE; steady decay points at uncertainty and drift.

Beside the generated frames, two reference lines on the same clips:
``vae``    the VAE round-trip of the ground truth, the best any DiT can reach;
``static`` frame 0 repeated, the trivial "nothing moves" prediction.

    python conditioning/scripts/diag_per_frame.py --tag T1 --num-clips 1000 --out-dir out/diag_T1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "third_party/Task_specific_JDM")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import encode_latents, stack_latent_stats
from f_toy.engine.dit_sample import sample
from f_toy.evaluation.checkpoints import load_vae

from conditioning.lib.dit_eval import condition_raw_frame_cutoff
from make_comparison import load_condition

PROJ = Path(__file__).resolve().parents[2]
CK = PROJ / "checkpoints/waymo"


def per_frame(pred, gt, lp):
    """(B,3,T,H,W) in [0,1] -> psnr (B,T), lpips (B,T)."""
    B, C, T, H, W = gt.shape
    mse = ((pred - gt) ** 2).mean(dim=(1, 3, 4)).clamp_min(1e-10)
    psnr = 10 * torch.log10(1.0 / mse)
    a = pred.transpose(1, 2).reshape(B * T, C, H, W)
    b = gt.transpose(1, 2).reshape(B * T, C, H, W)
    with torch.no_grad():
        l = lp(a, b, normalize=True).reshape(B, T)
    return psnr.cpu().numpy(), l.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="T1")
    ap.add_argument("--num-clips", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default="out/diag_T1")
    args = ap.parse_args()
    dev = "cuda"
    out = PROJ / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    vae, _ = load_vae(CK / "vae_50e24b", "best", dev)
    vae.eval()
    ls = torch.load(CK / "vae_50e24b/best.pt", map_location="cpu", weights_only=False)["latent_stats"]
    dit, conditioner, source, meta = load_condition(args.tag, CK, dev, "best")
    k = meta["fixed_k"]
    cutoff = condition_raw_frame_cutoff(k, vae.temporal_compression_factor)
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex", verbose=False).to(dev).eval()
    print(f"{args.tag}: epoch {meta['epoch']}, k={k}, cutoff raw {cutoff}, {args.steps} steps", flush=True)

    res = {}
    for split in ("train", "test"):
        ds = get_dataset_plugin({"DATASET": "waymo"}).DiskDataset(str(PROJ / f"data/driving_wp_f16/{split}"))
        # every n-th clip, so the subset spans the split's scenarios rather than the first few
        stride = max(1, len(ds) // args.num_clips)
        sub = Subset(ds, list(range(0, stride * args.num_clips, stride)))
        loader = DataLoader(sub, batch_size=args.batch, shuffle=False, num_workers=4)
        acc = {m: {"psnr": [], "lpips": []} for m in ("gen", "vae", "static")}
        torch.manual_seed(0)
        for videos, _ in loader:
            videos = videos.to(dev)
            B, _, T, H, W = videos.shape
            Tz = vae.latent_num_frames(T)
            mean, std = stack_latent_stats(ls, Tz, dev)
            with torch.no_grad():
                z = encode_latents(vae, videos, mean, std)
                ctx = None
                if source is not None:
                    cut = torch.full((B,), cutoff, device=dev, dtype=torch.long)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        r = source.encode(videos, cut)
                    ctx = conditioner(r.float())
                gen = sample(dit, vae, ls, tuple(z.shape), num_steps=args.steps, device=dev,
                             crossattn_emb=ctx, condition_latent=z, num_condition_frames=k,
                             target_frames=T)
                mu, _ = vae.encode(videos)
                rec = vae.decode(mu)[:, :, :T].clamp(0, 1)
                stat = videos[:, :, :1].expand_as(videos)
            for name, pred in (("gen", gen), ("vae", rec), ("static", stat)):
                p, l = per_frame(pred.float(), videos.float(), lp)
                acc[name]["psnr"].append(p)
                acc[name]["lpips"].append(l)
        res[split] = {m: {q: np.concatenate(v) for q, v in d.items()} for m, d in acc.items()}
        g = res[split]["gen"]
        print(f"{split}: {len(sub)} clips (every {stride}th). generated frames {cutoff}-{T - 1}: "
              f"PSNR {g['psnr'][:, cutoff:].mean():.3f}, LPIPS {g['lpips'][:, cutoff:].mean():.4f}", flush=True)

    np.savez(out / "per_frame.npz", **{f"{s}_{m}_{q}": res[s][m][q]
                                       for s in res for m in res[s] for q in res[s][m]})

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1a1a1a", "#55555a", "#dcdcd8"
    COL = {"train": "#2a78d6", "test": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    T = res["test"]["gen"]["psnr"].shape[1]
    x = np.arange(T)
    for ax, q, lab in ((axes[0], "psnr", "PSNR (dB), higher is better"),
                       (axes[1], "lpips", "LPIPS, lower is better")):
        for s in ("train", "test"):
            ax.plot(x, res[s]["gen"][q].mean(0), color=COL[s], lw=2.2, marker="o", ms=3.5,
                    label=f"{args.tag} generated, {s}")
            ax.plot(x, res[s]["vae"][q].mean(0), color=COL[s], lw=1.3, ls="--", alpha=0.8,
                    label=f"VAE round-trip, {s}")
            ax.plot(x, res[s]["static"][q].mean(0), color=COL[s], lw=1.3, ls=":", alpha=0.8,
                    label=f"frame 0 repeated, {s}")
        ax.axvspan(-0.5, cutoff - 0.5, color="#f0f0ec", zorder=0)
        ax.text(cutoff - 0.6, ax.get_ylim()[1], "context", ha="right", va="top", fontsize=8.5, color=MUTED)
        ax.set_xlabel("frame index (10 Hz)", color=MUTED)
        ax.set_title(lab, loc="left", fontsize=11, color=INK)
        ax.set_xlim(-0.5, T - 0.5)
        ax.grid(True, color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
    fig.suptitle(f"{args.tag} per-frame error, {len(res['test']['gen']['psnr'])} clips per split, "
                 f"{args.steps} Euler steps", x=0.06, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "per_frame.png", dpi=170, facecolor="white")

    summary = {s: {m: {q: [round(float(v), 4) for v in res[s][m][q].mean(0)] for q in ("psnr", "lpips")}
                   for m in res[s]} for s in res}
    (out / "per_frame.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out / "per_frame.png")


if __name__ == "__main__":
    main()
