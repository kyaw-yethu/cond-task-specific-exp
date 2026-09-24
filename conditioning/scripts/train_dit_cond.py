#!/usr/bin/env python
"""Train a MiniDiT from scratch with a driving representation as its
cross-attention conditioning.

`f_toy.engine.dit.train_dit` builds its own model and takes no conditioning
source, and `Task_specific_JDM` is read-only here, so the loop is reproduced.
Every numerical ingredient is imported from `f_toy` rather than rewritten, so
that condition **N** trained by this script is comparable to `dit10m32b100e`:
`sample_flow_matching_batch`, `sample_condition_mask`,
`apply_condition_mask_to_xt`, `replace_conditioned_velocity`,
`flow_matching_loss`, `build_dit`, `encode_latents`, `stack_latent_stats`,
`get_schedule` and `set_lr` all come from there. What is added is four lines:
recover each sample's raw-frame cutoff from the condition mask, encode that
sample's context window, and pass the tokens as `crossattn_emb`.

`--cond none` is condition **N** and falls through to `MiniDiT`'s own
`null_context`, so the floor and the conditioned runs share this one script.

Usage:
    python conditioning/scripts/train_dit_cond.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --cond drivejepa --cond-run drivejepa_T_vitl256 --tag T \\
        --ckpt-dir <proj>/checkpoints \\
        --dataset-dir <proj>/data/driving_wp_f16/train \\
        --val-dataset-dir <proj>/data/driving_wp_f16/test \\
        --config <proj>/configs/dit_cond.yaml
"""
import json
import sys
import time
from pathlib import Path


def _bootstrap():
    jdm = None
    for i, a in enumerate(sys.argv):
        if a == "--jdm" and i + 1 < len(sys.argv):
            jdm = sys.argv[i + 1]
        elif a.startswith("--jdm="):
            jdm = a.split("=", 1)[1]
    jdm = jdm or "third_party/Task_specific_JDM"
    if not Path(jdm, "f_toy").is_dir():
        sys.exit(f"no f_toy under {jdm}; pass --jdm")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    return jdm


JDM = _bootstrap()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from f_toy.config import build_arg_parser, get_config, get_device, set_seed
from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import build_dit, encode_latents, stack_latent_stats
from f_toy.evaluation.checkpoints import load_vae
from f_toy.schedule import get_schedule, set_lr
from f_toy.models.dit_loss import (apply_condition_mask_to_xt, flow_matching_loss,
                                   replace_conditioned_velocity, sample_condition_mask,
                                   sample_flow_matching_batch)

from conditioning.lib.cached import CachedCondDataset, read_cache_meta
from conditioning.lib.cond_dit import ConditionedDiT, CrossAttnConditioner, latent_cutoffs
from conditioning.lib.r_sources import build_cond_source

def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--cond", default="none",
                        choices=["none", "drivejepa", "vjepa2", "traj_vae"],
                        help="'none' is condition N: MiniDiT's own null_context")
    parser.add_argument("--cond-run", default=None,
                        help="checkpoints/<dataset>/<cond-run>/ holding the encoder")
    parser.add_argument("--cond-which", default="last")
    parser.add_argument("--context-width", type=int, default=6,
                        help="raw frames the encoder sees: the k=2 window, uniform "
                             "across the batch (see conditioning.lib.r_sources)")
    parser.add_argument("--cond-img", type=int, default=256)
    parser.add_argument("--tag", required=True, help="run name under checkpoints/<dataset>/")
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=200)
    parser.add_argument("--cache-dir", default=None,
                        help="a conditioning/scripts/build_cache.py directory. With it, k is FIXED at "
                             "the cache's value, the VAE and the r encoder are never run "
                             "during training, and the 17.7 GB of video is never read. "
                             "About 5.9x faster; see conditioning.lib.cached for what fixing k "
                             "costs")
    parser.add_argument("--val-every", type=int, default=1,
                        help="validate every N epochs")
    args = parser.parse_args()
    if not (args.dataset_dir and args.val_dataset_dir):
        parser.error("--dataset-dir and --val-dataset-dir are both required")

    # get_config merges f_toy's DEFAULT_CFG, the dataset defaults and the YAML, so
    # every key the imported helpers read (GRAD_CLIP, WARMUP_FRAC, DIT_*, FRAMES ...)
    # is populated exactly as train_dit.py would have it. That is what keeps a run
    # from this script comparable to dit10m32b100e.
    overrides = {k: v for k, v in (("EPOCHS", args.epochs), ("BATCH", args.batch),
                                   ("SEED", args.seed)) if v is not None}
    cfg = get_config(dataset=args.dataset, config_path=args.config, **overrides)
    device = args.device or get_device()
    seed = int(cfg.get("SEED", 0))
    set_seed(seed)

    ckpt_dir = Path(args.ckpt_dir) / cfg["DATASET"]
    run_dir = ckpt_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(run_dir / "train.log", "w")

    def log(line):
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    cached = args.cache_dir is not None
    vae = source = cache_meta = None
    fixed_k = None

    if cached:
        cache_meta = read_cache_meta(args.cache_dir)
        fixed_k = int(cache_meta["fixed_k"])
        spl = cache_meta["splits"]["train"]
        if args.cond != "none":
            if cache_meta["cond"] != args.cond:
                parser.error(f"cache holds {cache_meta['cond']!r}, asked for {args.cond!r}")
            cfg["DIT_CROSSATTN_DIM"] = int(spl["dim"])
            log(f"conditioning: {args.cond} ({cache_meta['cond_run']}) from cache, "
                f"grid {spl['grid']} -> {spl['tokens']} tokens of width {spl['dim']}")
        else:
            log("conditioning: none (condition N, MiniDiT's learned null_context)")
        log(f"cache {args.cache_dir}: k FIXED at {fixed_k}, VAE and encoder not run "
            f"during training")
        # The encoder window has to travel with the checkpoint: eval rebuilds the
        # source from these, and a width that differs from the cache's would feed
        # the conditioner a different token count or different frames.
        cfg["COND_CONTEXT_WIDTH"] = int(cache_meta["context_width"])
        cfg["COND_IMG"] = int(cache_meta["cond_img"])
    else:
        # ---- frozen VAE ----
        vae, _ = load_vae(ckpt_dir / args.vae_run, args.vae_which, device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        latent_stats = torch.load(ckpt_dir / args.vae_run / f"{args.vae_which}.pt",
                                  map_location="cpu", weights_only=False)["latent_stats"]

        # ---- conditioning source ----
        if args.cond != "none":
            if not args.cond_run:
                parser.error("--cond-run is required unless --cond none")
            source = build_cond_source(args.cond, ckpt_dir, device,
                                       context_width=args.context_width,
                                       img_size=args.cond_img,
                                       run=args.cond_run, which=args.cond_which)
            cfg["DIT_CROSSATTN_DIM"] = source.dim
            cfg["COND_CONTEXT_WIDTH"] = args.context_width
            cfg["COND_IMG"] = args.cond_img
            log(f"conditioning: {args.cond} ({args.cond_run}) grid {source.grid} "
                f"-> {source.num_tokens} tokens of width {source.dim}")
        else:
            log("conditioning: none (condition N, MiniDiT's learned null_context)")

    # ---- model ----
    core_dit = build_dit(cfg).to(device)
    if cached and args.cond != "none":
        spl = cache_meta["splits"]["train"]
        conditioner = CrossAttnConditioner(int(spl["dim"]), int(spl["tokens"])).to(device)
    elif source is not None:
        conditioner = CrossAttnConditioner(source.dim, source.num_tokens).to(device)
    else:
        conditioner = None
    model = ConditionedDiT(core_dit, conditioner).to(device)
    n_dit = sum(p.numel() for p in core_dit.parameters())
    n_cond = sum(p.numel() for p in conditioner.parameters()) if conditioner else 0
    log(f"DiT width {cfg['DIT_MODEL_CHANNELS']} x "
        f"{cfg['DIT_NUM_BLOCKS']} blocks")
    log(f"DiT {n_dit:,} params + conditioner {n_cond:,} = {n_dit + n_cond:,} trainable")

    # ---- data ----
    if cached:
        with_r = args.cond != "none"
        train_ds = CachedCondDataset(args.cache_dir, "train", with_r=with_r)
        val_ds = CachedCondDataset(args.cache_dir, "val", with_r=with_r)
    else:
        plugin = get_dataset_plugin(cfg)
        train_ds = plugin.DiskDataset(args.dataset_dir)
        val_ds = plugin.DiskDataset(args.val_dataset_dir)
    batch = int(cfg.get("BATCH", 16))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=args.workers)
    log(f"train {len(train_ds)} clips, val {len(val_ds)} clips, batch {batch}")

    if cached:
        tc = None
        min_cond = max_cond = fixed_k     # a fixed mask, not a per-sample draw
    else:
        Tz = vae.latent_num_frames(cfg["FRAMES"])
        tc = vae.temporal_compression_factor
        mean, std = stack_latent_stats(latent_stats, Tz, device)
        min_cond, max_cond = cfg["DIT_MIN_COND_FRAMES"], cfg["DIT_MAX_COND_FRAMES"]
    cfg["DIT_FIXED_K"] = fixed_k          # recorded in the checkpoint

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["DIT_LR"],
                            weight_decay=cfg.get("DIT_WD", 0.0))
    epochs = int(cfg["EPOCHS"])
    total = epochs * len(train_loader)
    sched = get_schedule(cfg, "DIT_LR")
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def step_loss(batch_in):
        """One flow-matching loss. Cached: `batch_in` is already (x0, r).
        Uncached: it is (videos, labels) and the VAE and encoder run here."""
        if cached:
            x0, r = batch_in
            x0 = x0.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True) if torch.is_tensor(r) and r.ndim == 3 else None
        else:
            videos = batch_in[0].to(device, non_blocking=True)
            x0, r = encode_latents(vae, videos, mean, std), None
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            xt, t, v_target = sample_flow_matching_batch(x0)
            cond_mask = sample_condition_mask(x0, min_cond, max_cond)
            xt = apply_condition_mask_to_xt(xt, x0, cond_mask)
            if not cached and source is not None:
                r = source.encode(videos, latent_cutoffs(cond_mask, tc))
            v_pred = model(xt, t, r_tokens=r, condition_mask_B_1_T_H_W=cond_mask)
            v_pred = replace_conditioned_velocity(v_pred, v_target, cond_mask)
            return flow_matching_loss(v_pred, v_target)

    step, best_val, hist = 0, float("inf"), []
    for ep in range(1, epochs + 1):
        model.train()
        tr, n_steps = 0.0, len(train_loader)
        t0 = time.time()
        for i, batch_in in enumerate(train_loader):
            lr = sched(step, total, cfg["DIT_LR"], cfg.get("DIT_LR_END", 1e-6),
                       cfg["WARMUP_FRAC"])
            set_lr(opt, lr)
            loss = step_loss(batch_in)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("GRAD_CLIP", 1.0))
            scaler.step(opt)
            scaler.update()
            tr += loss.item()
            step += 1
            if (i + 1) % args.log_every_steps == 0 or (i + 1) == n_steps:
                el = time.time() - t0
                log(f"  step {i + 1:5d}/{n_steps} | {el:5.0f}s | "
                    f"~{el / (i + 1) * (n_steps - i - 1):5.0f}s left | "
                    f"{el / (i + 1) * 1000:.0f}ms/step | lr {lr:.2e}")

        tr = tr / n_steps
        if ep % args.val_every == 0 or ep == epochs:
            model.eval()
            va = 0.0
            with torch.no_grad():
                for batch_in in val_loader:
                    va += step_loss(batch_in).item()
            va /= len(val_loader)
        else:
            va = float("nan")
        hist.append(dict(epoch=ep, train_flow=tr, val_flow=va, secs=time.time() - t0))
        log(f"[DiT-{args.tag}] ep {ep:>3}/{epochs} train_flow {tr:.6f} "
            f"val_flow {va:.6f} ({time.time() - t0:.0f}s)")

        payload = dict(cfg=cfg, model=core_dit.state_dict(),
                       conditioner=(conditioner.state_dict() if conditioner else None),
                       cond=args.cond, cond_run=args.cond_run,
                       # In the cached path `source` is never built, so these come
                       # from the cache metadata instead of the encoder.
                       cond_grid=(source.grid if source is not None else
                                  (cache_meta["splits"]["train"]["grid"] if cached else None)),
                       cond_dim=(source.dim if source is not None else
                                 (cache_meta["splits"]["train"]["dim"] if cached else None)),
                       epoch=ep, seed=seed, val_flow=va,
                       VAE_RUN=args.vae_run, VAE_WHICH=args.vae_which)
        torch.save(payload, run_dir / "last.pt")
        if va == va and va < best_val:      # NaN on a skipped validation epoch
            best_val, payload["best_epoch"] = va, ep
            torch.save(payload, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(hist, indent=2))

    log(f"done. best val_flow {best_val:.6f}. checkpoints in {run_dir}")
    log_fh.close()


if __name__ == "__main__":
    main()
