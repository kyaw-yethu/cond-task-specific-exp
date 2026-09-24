#!/usr/bin/env python
"""Train a causal video VAE on womd_7hz_f33, MiniWan from scratch or Wan2.1 from its release.

Both candidates are the same architecture class, ``f_toy.models.vae.MiniWan``;
only the width and the starting weights differ:

``miniwan``  MiniWan(dim=20), 5.6M params, trained from scratch: random init, input
             normalised with the train split pixel statistics.
``wan21``    MiniWan(dim=96), initialised from the released Wan2.1 VAE (127M params).
             Wan2.1 works on [-1, 1]; MiniWan on [0, 1] in and out. The input side
             becomes MiniWan's own InputNorm with mean 0.5 and std 0.5, and the
             output side, y -> (y + 1) / 2, is folded exactly into the decoder's last
             conv. The result is an ordinary MiniWan checkpoint, so ``load_vae``, the
             latent cache and the DiT sampler use it unchanged.

Everything else is identical between the two runs: data, loss (MSE + KL with
warm-up + LPIPS, as in vae_50e24b), optimiser, schedule, precision, effective
batch, epochs, validation subset and checkpoint selection. After training, the
per-channel latent statistics (frame 0 and the rest) are computed from train
clips and attached to both checkpoints, as scripts/train_vae.py does.

    torchrun --nproc_per_node 2 conditioning/scripts/train_vae_ft.py --model wan21 --run-name vae_wan21
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "third_party/Task_specific_JDM")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from f_toy.models.vae import MiniWan
from f_toy.models.vae_loss import kl_warmup, lpips_loss, vae_loss

PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data/womd_7hz_f33"
WAN_INIT = PROJ / "checkpoints/wan21_vae/Wan2.1_VAE.pth"


class Clips(Dataset):
    """uint8 videos.npy memmap -> float (3, T, H, W) in [0, 1]. Labels are not needed."""

    def __init__(self, split):
        self.path = DATA / split / "videos.npy"
        self.n = int(np.load(self.path, mmap_mode="r").shape[0])
        self.mm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self.mm[i])).float().div_(255.0)


def build(model, dropout):
    """Returns (MiniWan, arch cfg) with the pretrained weights loaded."""
    if model == "miniwan":
        # from scratch: random init, input normalised with the train split pixel statistics
        ps = json.loads((DATA / "train/pixel_stats.json").read_text())
        arch = dict(VAE_ZDIM=16, VAE_DIM=20, VAE_NUM_RES_BLOCKS=2, VAE_INPUT_NORM=True,
                    INPUT_NORM_MEAN=ps["mean"], INPUT_NORM_STD=ps["std"])
        m = MiniWan(z_dim=16, dim=20, num_res_blocks=2, dropout=dropout, input_norm=True,
                    norm_stats=(ps["mean"], ps["std"]))
        return m, arch

    arch = dict(VAE_ZDIM=16, VAE_DIM=96, VAE_NUM_RES_BLOCKS=2, VAE_INPUT_NORM=True,
                INPUT_NORM_MEAN=[0.5, 0.5, 0.5], INPUT_NORM_STD=[0.5, 0.5, 0.5])
    m = MiniWan(z_dim=16, dim=96, num_res_blocks=2, dropout=dropout, input_norm=True,
                norm_stats=([0.5] * 3, [0.5] * 3))
    sd = torch.load(WAN_INIT, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("input_norm.")]
    if missing or unexpected:
        raise RuntimeError(f"Wan2.1 into MiniWan(dim=96): missing {missing[:5]}, unexpected {unexpected[:5]}")
    head = m.decoder.head[2]
    with torch.no_grad():                     # y in [-1, 1]  ->  (y + 1) / 2 in [0, 1]
        head.weight.mul_(0.5)
        head.bias.mul_(0.5).add_(0.5)
    return m, arch


def cosine(step, total, base, end, warm_frac):
    w = max(1, int(total * warm_frac))
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return end + 0.5 * (base - end) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(core, loader, lp, dev):
    """Mean MSE, PSNR and LPIPS over a loader, plus per-clip LPIPS."""
    core.eval()
    mse_all, lp_all = [], []
    for x in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r, _, _ = core(x)
        r = r.float().clamp(0, 1)
        B, C, T, H, W = x.shape
        mse_all.append(((r - x) ** 2).mean(dim=(1, 2, 3, 4)).cpu())
        a = r.transpose(1, 2).reshape(B * T, C, H, W)
        b = x.transpose(1, 2).reshape(B * T, C, H, W)
        lp_all.append(lp(a, b, normalize=True).reshape(B, T).mean(1).cpu())
    mse = torch.cat(mse_all)
    lpc = torch.cat(lp_all)
    return dict(mse=float(mse.mean()), psnr=float((10 * torch.log10(1 / mse.clamp_min(1e-10))).mean()),
                lpips=float(lpc.mean())), lpc.numpy()


@torch.no_grad()
def latent_stats(core, ds, n, dev, batch):
    core.eval()
    sub = Subset(ds, list(np.linspace(0, len(ds) - 1, n).astype(int)))
    m0, mr = [], []
    for x in DataLoader(sub, batch_size=batch, num_workers=4):
        mu, _ = core.encode(x.to(dev))
        m0.append(mu[:, :, :1].float().cpu())
        mr.append(mu[:, :, 1:].float().cpu())
    m0, mr = torch.cat(m0), torch.cat(mr)
    return {"frame0": dict(mean=m0.mean(dim=(0, 2, 3, 4)), std=m0.std(dim=(0, 2, 3, 4))),
            "rest": dict(mean=mr.mean(dim=(0, 2, 3, 4)), std=mr.std(dim=(0, 2, 3, 4)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["miniwan", "wan21"], required=True)
    ap.add_argument("--run-name", dest="run", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=12, help="per rank")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-end", type=float, default=1e-6)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--kl-w", type=float, default=3e-4)
    ap.add_argument("--kl-warm-frac", type=float, default=0.1)
    ap.add_argument("--lpips-w", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--val-clips", type=int, default=500)
    ap.add_argument("--stats-clips", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    main_rank = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = PROJ / "checkpoints/womd" / args.run
    if main_rank:
        run_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(run_dir / "train.log", "a") if main_rank else None

    def log(s):
        if main_rank:
            print(s, flush=True)
            log_fh.write(s + "\n")
            log_fh.flush()

    core, arch = build(args.model, args.dropout)
    core = core.to(dev)
    n_params = sum(p.numel() for p in core.parameters())
    vae = DDP(core, device_ids=[local], output_device=local) if world > 1 else core

    train_ds = Clips("train")
    val_ds = Clips("val")
    val_sub = Subset(val_ds, list(range(0, len(val_ds), max(1, len(val_ds) // args.val_clips)))[:args.val_clips])
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=args.seed, drop_last=True) \
        if world > 1 else None
    loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=8, num_workers=4)

    eff = args.batch * world * args.accum
    steps_per_epoch = len(loader) // args.accum
    total = args.epochs * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    cfg = dict(arch, VAE_DROPOUT=args.dropout, VAE_STREAMING=False, FRAMES=33, IMG=96, FPS=7,
               DATASET="womd_7hz_f33", INIT=args.model, EPOCHS=args.epochs, EFFECTIVE_BATCH=eff,
               VAE_LR=args.lr, VAE_LR_END=args.lr_end, WARMUP_FRAC=args.warmup_frac, VAE_WD=args.wd,
               KL_W=args.kl_w, KL_WARM_FRAC=args.kl_warm_frac, VAE_LPIPS_W=args.lpips_w,
               VAE_LPIPS_NET="alex", GRAD_CLIP=args.grad_clip, PRECISION="bf16", SEED=args.seed)
    log(f"== {args.run}: {args.model}, {n_params / 1e6:.2f}M params, world {world}, per-rank batch "
        f"{args.batch} x accum {args.accum} = effective {eff}; {steps_per_epoch} steps/epoch, "
        f"{total} steps total")
    log(json.dumps(cfg))

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.wd)
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    kl_warm = max(1, int(total * args.kl_warm_frac))

    best, step, hist = float("inf"), 0, []
    if main_rank:
        m0, _ = evaluate(core, val_loader, lp, dev)
        log(f"[{args.run}] ep   0 (init) val PSNR {m0['psnr']:.3f} LPIPS {m0['lpips']:.4f} MSE {m0['mse']:.6f}")
        hist.append(dict(epoch=0, **{f"val_{k}": v for k, v in m0.items()}))
    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        vae.train()
        t0 = time.time()
        acc = dict(rec=0.0, kl=0.0, lp=0.0, n=0)
        opt.zero_grad(set_to_none=True)
        for i, x in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon, mu, lv = vae(x)
            recon, mu, lv = recon.float(), mu.float(), lv.float()
            loss, rec, kl = vae_loss(recon, x, mu, lv, kl_warmup(step, kl_warm, args.kl_w))
            l_lp = lpips_loss(lp, recon.clamp(0.0, 1.0), x)
            loss = (loss + args.lpips_w * l_lp) / args.accum
            loss.backward()
            acc["rec"] += rec.item(); acc["kl"] += kl.item(); acc["lp"] += l_lp.item(); acc["n"] += 1
            if (i + 1) % args.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = cosine(step, total, args.lr, args.lr_end, args.warmup_frac)
            nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if main_rank and (step % 100 == 0 or step == 1):
                el = time.time() - t0
                done = (i + 1) // args.accum
                log(f"  step {step} ({done}/{steps_per_epoch} this epoch) | {el / done * 1000:.0f} ms/step | "
                    f"~{el / done * (steps_per_epoch - done) / 60:.0f} min left in epoch | "
                    f"rec {acc['rec'] / acc['n']:.5f} lpips {acc['lp'] / acc['n']:.4f} lr {opt.param_groups[0]['lr']:.2e}")
            if args.max_steps and step >= args.max_steps:
                break
        if main_rank:
            n = max(1, acc["n"])
            line = (f"[{args.run}] ep {ep:3d}/{args.epochs} train rec {acc['rec'] / n:.6f} "
                    f"LPIPS {acc['lp'] / n:.4f} KL {acc['kl'] / n:.3f} ({time.time() - t0:.0f}s)")
            rec_ = dict(epoch=ep, train_rec=acc["rec"] / n, train_lpips=acc["lp"] / n, train_kl=acc["kl"] / n)
            if ep % args.val_every == 0 or ep == args.epochs or args.max_steps:
                m, _ = evaluate(core, val_loader, lp, dev)
                score = m["mse"] + args.lpips_w * m["lpips"]
                line += f" | val PSNR {m['psnr']:.3f} LPIPS {m['lpips']:.4f} MSE {m['mse']:.6f}"
                rec_.update({f"val_{k}": v for k, v in m.items()})
                ck = dict(state_dict=core.state_dict(), cfg=cfg, epoch=ep, val=m)
                torch.save(ck, run_dir / "last.pt")
                if score < best:
                    best = score
                    torch.save(ck, run_dir / "best.pt")
                    line += "  *best*"
            log(line)
            hist.append(rec_)
            (run_dir / "history.json").write_text(json.dumps(hist, indent=1))
        if world > 1:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break

    if main_rank:
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        core.load_state_dict(ck["state_dict"])
        stats = latent_stats(core, train_ds, args.stats_clips, dev, 8)
        for which in ("best", "last"):
            c = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=False)
            c["latent_stats"] = stats
            torch.save(c, run_dir / f"{which}.pt")
        log(f"latent stats from {args.stats_clips} train clips (best weights): frame0 std "
            f"{stats['frame0']['std'].mean():.4f}, rest std {stats['rest']['std'].mean():.4f}")
        full = DataLoader(val_ds, batch_size=8, num_workers=6)
        m, per_clip = evaluate(core, full, lp, dev)
        np.save(run_dir / "val_lpips_per_clip.npy", per_clip)
        (run_dir / "val_full.json").write_text(json.dumps(dict(epoch=ck["epoch"], **m), indent=1))
        log(f"[{args.run}] best (epoch {ck['epoch']}) on all {len(val_ds)} val clips: "
            f"PSNR {m['psnr']:.3f} LPIPS {m['lpips']:.4f} MSE {m['mse']:.6f}")
        log("DONE")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
