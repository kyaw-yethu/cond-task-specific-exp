#!/usr/bin/env python
"""Train Traj-VAE v2 on womd_7hz_f33 (TRAJ_VAE_PLAN.md).

Encoder cloned from a trained MiniWan, heads on top, k drawn from 1..3 per batch.
Validation reports waypoint ADE at k = 1, 2 and 3 separately, which is the number
V1 failed on at k = 1.

    torchrun --nproc_per_node 4 scripts/train_traj_v2.py --run-name traj_v2 --epochs 30
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "third_party/Task_specific_JDM")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from f_toy.models.vae import MiniWan
from cond_eval.traj_v2 import FUT, TrajV2, chunk_end, motion_targets, traj_losses, waypoint_targets

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data/womd_7hz_f33"
VAE_RUN = PROJ / "checkpoints/womd/vae_miniwan_womd/best.pt"


class ClipsWithTargets(Dataset):
    def __init__(self, split):
        self.vpath = DATA / split / "videos.npy"
        self.n = int(np.load(self.vpath, mmap_mode="r").shape[0])
        t = np.load(DATA / split / "traj_targets.npz")
        self.t = {k: t[k] for k in t.files}
        self.mm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(self.vpath, mmap_mode="r")
        v = torch.from_numpy(np.array(self.mm[i])).float().div_(255.0)
        t = self.t
        pos_all = np.concatenate([t["ego_position"][i], t["future_position"][i]], 0)
        yaw_all = np.concatenate([t["ego_heading"][i], t["future_heading"][i]], 0)
        val_all = np.concatenate([np.ones(t["ego_position"].shape[1], np.float32), t["future_valid"][i]], 0)
        return (v, torch.from_numpy(t["ego_position"][i]), torch.from_numpy(t["ego_heading"][i]),
                torch.from_numpy(t["ego_speed"][i]), torch.from_numpy(pos_all), torch.from_numpy(yaw_all),
                torch.from_numpy(val_all), torch.from_numpy(t["occupancy"][i]).float(),
                torch.from_numpy(t["lane_curvature"][i]), torch.from_numpy(t["in_intersection"][i]),
                torch.from_numpy(t["tl_state"][i].astype(np.int64)))


def targets_for(batch, ends, k, dev):
    v, pos, yaw, spd, pos_all, yaw_all, val_all, occ, curv, inter, tl = [x.to(dev, non_blocking=True) for x in batch]
    cutoff = 1 + (k - 1) * 4
    way, way_valid = waypoint_targets(pos_all, yaw_all, val_all, cutoff)
    tgt = dict(motion=motion_targets(pos, yaw, spd, ends), way=way, way_valid=way_valid,
               occ=occ[:, ends], curvature=curv[:, ends], intersection=inter[:, ends],
               tl=(tl[:, ends] + 1).clamp(0, 4))
    return v, tgt


def cosine(step, total, base, end, warm_frac):
    w = max(1, int(total * warm_frac))
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return end + 0.5 * (base - end) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def validate(model, loader, ends, weights, kl_w, dev):
    model.eval()
    tot, parts, n = 0.0, {}, 0
    ade = {1: [], 2: [], 3: []}
    for batch in loader:
        for k in (1, 2, 3):
            v, tgt = targets_for(batch, ends, k, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(v, k)
            out = {a: (b.float() if torch.is_tensor(b) else b) for a, b in out.items()}
            loss, p = traj_losses(out, tgt, weights, kl_w)
            pick = out["way_logit"].argmax(1)
            chosen = out["way"][torch.arange(len(pick), device=dev), pick]
            d = ((chosen - tgt["way"]) ** 2).sum(-1).sqrt()
            ade[k].append(((d * tgt["way_valid"]).sum(-1) / tgt["way_valid"].sum(-1).clamp_min(1)).cpu())
            if k == 2:
                tot += float(loss); n += 1
                for a, b in p.items():
                    parts[a] = parts.get(a, 0.0) + b
    return (tot / max(n, 1), {a: b / max(n, 1) for a, b in parts.items()},
            {k: float(torch.cat(v).mean()) for k, v in ade.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", dest="run", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--enc-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lr-end-frac", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--kl-w", type=float, default=1e-4)
    ap.add_argument("--w-motion", type=float, default=1.0)
    ap.add_argument("--w-way", type=float, default=1.0)
    ap.add_argument("--w-occ", type=float, default=0.5)
    ap.add_argument("--w-scene", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-clips", type=int, default=500)
    ap.add_argument("--stats-clips", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    main_rank = rank == 0
    torch.manual_seed(args.seed)
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

    ck = torch.load(VAE_RUN, map_location="cpu", weights_only=False)
    vcfg = ck["cfg"]
    vae = MiniWan(z_dim=vcfg["VAE_ZDIM"], dim=vcfg["VAE_DIM"], num_res_blocks=vcfg["VAE_NUM_RES_BLOCKS"],
                  dropout=args.dropout, input_norm=vcfg.get("VAE_INPUT_NORM", False))
    vae.load_state_dict(ck["state_dict"])
    model = TrajV2(vae, dropout=args.dropout).to(dev)
    net = DDP(model, device_ids=[local], output_device=local) if world > 1 else model
    n_enc = sum(p.numel() for p in model.backbone_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())

    train_ds = ClipsWithTargets("train")
    val_ds = ClipsWithTargets("val")
    val_sub = Subset(val_ds, list(range(0, len(val_ds), max(1, len(val_ds) // args.val_clips)))[:args.val_clips])
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=8, num_workers=4)

    ends = chunk_end(33)
    weights = dict(motion=args.w_motion, way=args.w_way, occ=args.w_occ, scene=args.w_scene)
    opt = torch.optim.AdamW([
        dict(params=model.backbone_parameters(), lr=args.enc_lr),
        dict(params=model.head_parameters(), lr=args.head_lr)], weight_decay=args.wd)
    steps_per_epoch = len(loader)
    total = args.epochs * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    cfg = dict(vars(args), VAE_RUN=str(VAE_RUN), VAE_ZDIM=vcfg["VAE_ZDIM"], VAE_DIM=vcfg["VAE_DIM"],
               VAE_NUM_RES_BLOCKS=vcfg["VAE_NUM_RES_BLOCKS"], VAE_INPUT_NORM=vcfg.get("VAE_INPUT_NORM", False),
               FRAMES=33, IMG=96, FPS=7, DATASET="womd_7hz_f33", FUT=FUT)
    log(f"== {args.run}: encoder {n_enc / 1e6:.2f}M (from {VAE_RUN.parent.name}), heads {n_head / 1e6:.2f}M; "
        f"world {world}, per-rank batch {args.batch} (effective {args.batch * world}); "
        f"{steps_per_epoch} steps/epoch, {total} total")
    log(json.dumps({k: v for k, v in cfg.items() if not isinstance(v, Path)}))

    best, step, hist = float("inf"), 0, []
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t0, acc, nb = time.time(), {}, 0
        for batch in loader:
            k = int(rng.integers(1, 4))
            v, tgt = targets_for(batch, ends, k, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(v, k)
            out = {a: (b.float() if torch.is_tensor(b) else b) for a, b in out.items()}
            loss, parts = traj_losses(out, tgt, weights, args.kl_w)
            for g, lr in zip(opt.param_groups, (args.enc_lr, args.head_lr)):
                g["lr"] = cosine(step, total, lr, lr * args.lr_end_frac, args.warmup_frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1; nb += 1
            for a, b in parts.items():
                acc[a] = acc.get(a, 0.0) + b
            if main_rank and (step % 200 == 0 or step == 1):
                el = time.time() - t0
                log(f"  step {step} ({nb}/{steps_per_epoch}) | {el / nb * 1000:.0f} ms/step | "
                    f"~{el / nb * (steps_per_epoch - nb) / 60:.0f} min left in epoch | "
                    + " ".join(f"{a} {b / nb:.4f}" for a, b in acc.items()))
            if args.max_steps and step >= args.max_steps:
                break
        if main_rank:
            vl, vp, ade = validate(model, val_loader, ends, weights, args.kl_w, dev)
            line = (f"[{args.run}] ep {ep:3d}/{args.epochs} train " +
                    " ".join(f"{a} {b / max(nb, 1):.4f}" for a, b in acc.items()) +
                    f" | val {vl:.4f} ADE k1 {ade[1]:.3f} k2 {ade[2]:.3f} k3 {ade[3]:.3f} ({time.time() - t0:.0f}s)")
            c = dict(state_dict=model.state_dict(), cfg=cfg, epoch=ep, val_loss=vl, val_parts=vp, val_ade=ade)
            torch.save(c, run_dir / "last.pt")
            if vl < best:
                best = vl
                torch.save(c, run_dir / "best.pt")
                line += "  *best*"
            log(line)
            hist.append(dict(epoch=ep, val_loss=vl, val_ade=ade, **{f"train_{a}": b / max(nb, 1) for a, b in acc.items()}))
            (run_dir / "history.json").write_text(json.dumps(hist, indent=1))
        if world > 1:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break

    if main_rank:
        c = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(c["state_dict"])
        model.eval()
        sub = Subset(train_ds, list(np.linspace(0, len(train_ds) - 1, args.stats_clips).astype(int)))
        m0, mr = [], []
        with torch.no_grad():
            for batch in DataLoader(sub, batch_size=8, num_workers=4):
                mu = model.latent_repr(batch[0].to(dev))
                m0.append(mu[:, :, :1].float().cpu()); mr.append(mu[:, :, 1:].float().cpu())
        m0, mr = torch.cat(m0), torch.cat(mr)
        stats = {"frame0": dict(mean=m0.mean(dim=(0, 2, 3, 4)), std=m0.std(dim=(0, 2, 3, 4))),
                 "rest": dict(mean=mr.mean(dim=(0, 2, 3, 4)), std=mr.std(dim=(0, 2, 3, 4)))}
        for which in ("best", "last"):
            cc = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=False)
            cc["traj_stats"] = stats
            torch.save(cc, run_dir / f"{which}.pt")
        log(f"traj stats from {args.stats_clips} train clips: frame0 std {stats['frame0']['std'].mean():.4f}, "
            f"rest std {stats['rest']['std'].mean():.4f}")
        log(f"[{args.run}] best epoch {c['epoch']}: val {c['val_loss']:.4f}, ADE "
            f"k1 {c['val_ade'][1]:.3f} k2 {c['val_ade'][2]:.3f} k3 {c['val_ade'][3]:.3f}")
        log("DONE")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
