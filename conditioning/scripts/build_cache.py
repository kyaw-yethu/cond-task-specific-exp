#!/usr/bin/env python
"""Precompute the VAE latents and the conditioning tokens for a fixed k.

Writes, per split, two fp16 `.npy` files opened later as memmaps:

    x0_<split>.npy    (N, VAE_ZDIM, Tz, Hz, Wz)   ~921 MB for 40,000 clips
    r_<split>_<tag>.npy  (N, tokens, dim)         ~62.9 GB for ViT-L at 768x1024

Both are pure functions of the clip once k is fixed, so this runs once per
encoder instead of every epoch. See `conditioning.lib.cached` for why k is fixed and
what it costs.

`--fixed-k` picks k (default 2) and `--context-width` the raw frames the encoder
sees. The width has to cover the k context frames, so 6 at k=2 (raw 0..4, the
last repeated for the tubelet) and 2 at k=1 for the ViTs (raw 0 twice). The
causal Traj-VAE takes 1 at k=1, raw 0 alone, since it has no tubelet to fill.

Disk check before starting, because the token cache is most of the free space.
Delete one encoder's cache before building the next.

Usage:
    python conditioning/scripts/build_cache.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --ckpt-dir <proj>/checkpoints --cache-dir <proj>/cache \\
        --cond drivejepa --cond-run drivejepa_T_vitl256 --tag T \\
        --dataset-dir <proj>/data/driving_wp_f16/train \\
        --val-dataset-dir <proj>/data/driving_wp_f16/test
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
    jdm = jdm or "third_party/Task_specific_JDM"
    if not Path(jdm, "f_toy").is_dir():
        sys.exit(f"no f_toy under {jdm}")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    return jdm


JDM = _bootstrap()

import numpy as np
import torch
from torch.utils.data import DataLoader

from f_toy.config import build_arg_parser, get_config, get_device
from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import encode_latents, stack_latent_stats
from f_toy.evaluation.checkpoints import load_vae

from conditioning.lib.cached import read_cache_meta, write_cache_meta
from conditioning.lib.dit_eval import condition_raw_frame_cutoff
from conditioning.lib.r_sources import build_cond_source


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 1024 ** 3


def build_split(name, dataset_dir, vae, mean, std, source, cache_dir, cfg, device,
                batch, workers, tag, fixed_k):
    plugin = get_dataset_plugin(cfg)
    ds = plugin.DiskDataset(dataset_dir)
    n = len(ds)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

    Tz = vae.latent_num_frames(cfg["FRAMES"])
    Hz = Wz = cfg["IMG"] // vae.spatial_compression_factor
    zdim = cfg["VAE_ZDIM"]
    cutoff = condition_raw_frame_cutoff(fixed_k, vae.temporal_compression_factor)

    x0_file = f"x0_{name}.npy"
    x0_path = Path(cache_dir) / x0_file
    x0_mm = np.lib.format.open_memmap(x0_path, mode="w+", dtype=np.float16,
                                      shape=(n, zdim, Tz, Hz, Wz))
    r_file = r_mm = None
    if source is not None:
        r_file = f"r_{name}_{tag}.npy"
        r_mm = np.lib.format.open_memmap(Path(cache_dir) / r_file, mode="w+",
                                         dtype=np.float16,
                                         shape=(n, source.num_tokens, source.dim))
    gb = (x0_mm.nbytes + (r_mm.nbytes if r_mm is not None else 0)) / 1024 ** 3
    print(f"[{name}] {n} clips -> {gb:.2f} GB (free {free_gb(cache_dir):.1f} GB)")

    cut = torch.full((batch,), cutoff, device=device, dtype=torch.long)
    i, t0 = 0, time.time()
    with torch.no_grad():
        for videos, _ in loader:
            videos = videos.to(device, non_blocking=True)
            b = videos.shape[0]
            x0_mm[i:i + b] = encode_latents(vae, videos, mean, std).half().cpu().numpy()
            if r_mm is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    r = source.encode(videos, cut[:b])
                r_mm[i:i + b] = r.half().cpu().numpy()
            i += b
            if i % (batch * 100) == 0 or i == n:
                el = time.time() - t0
                print(f"  {i}/{n}  {el:5.0f}s  ~{el / i * (n - i):5.0f}s left", flush=True)
    x0_mm.flush()
    if r_mm is not None:
        r_mm.flush()
    return dict(n=n, x0_file=x0_file, r_file=r_file,
                grid=(list(source.grid) if source else None),
                tokens=(source.num_tokens if source else None),
                dim=(source.dim if source else None))


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cond", default="none",
                        choices=["none", "drivejepa", "vjepa2", "traj_vae"])
    parser.add_argument("--cond-run", default=None)
    parser.add_argument("--cond-which", default="last")
    parser.add_argument("--fixed-k", type=int, default=2,
                        help="latent context frames, fixed for every clip")
    parser.add_argument("--context-width", type=int, default=6)
    parser.add_argument("--cond-img", type=int, default=256)
    parser.add_argument("--tag", default="N")
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not (args.dataset_dir and args.val_dataset_dir):
        parser.error("--dataset-dir and --val-dataset-dir are both required")

    cfg = get_config(dataset=args.dataset, config_path=args.config)
    need_w = condition_raw_frame_cutoff(args.fixed_k, 4)
    if args.cond != "none" and args.context_width < need_w:
        parser.error(f"--context-width {args.context_width} is narrower than the "
                     f"{need_w} raw context frames k={args.fixed_k} gives")
    device = args.device or get_device()
    batch = int(args.batch or 16)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir) / cfg["DATASET"]
    vae, _ = load_vae(ckpt_dir / args.vae_run, args.vae_which, device)
    vae.eval()
    latent_stats = torch.load(ckpt_dir / args.vae_run / f"{args.vae_which}.pt",
                              map_location="cpu", weights_only=False)["latent_stats"]
    Tz = vae.latent_num_frames(cfg["FRAMES"])
    mean, std = stack_latent_stats(latent_stats, Tz, device)

    source = None
    if args.cond != "none":
        source = build_cond_source(args.cond, ckpt_dir, device,
                                   context_width=args.context_width,
                                   img_size=args.cond_img,
                                   run=args.cond_run, which=args.cond_which)
        need = 40000 * source.num_tokens * source.dim * 2 / 1024 ** 3
        print(f"source {args.cond} ({args.cond_run}): {source.num_tokens} tokens "
              f"x {source.dim}, train cache ~{need:.1f} GB, free "
              f"{free_gb(cache_dir):.1f} GB")
        if free_gb(cache_dir) < need * 1.15:
            sys.exit(f"not enough free space; delete another encoder's cache first")

    splits = {}
    for name, d in (("train", args.dataset_dir), ("val", args.val_dataset_dir)):
        splits[name] = build_split(name, d, vae, mean, std, source, cache_dir, cfg,
                                   device, batch, args.workers, args.tag, args.fixed_k)

    write_cache_meta(cache_dir, fixed_k=args.fixed_k, tag=args.tag, cond=args.cond,
                     cond_run=args.cond_run, context_width=args.context_width,
                     cond_img=args.cond_img, frames=cfg["FRAMES"], img=cfg["IMG"],
                     vae_run=args.vae_run, vae_which=args.vae_which, splits=splits)
    print(f"\nwrote {cache_dir}/cache_meta.json")
    print(json.dumps(read_cache_meta(cache_dir), indent=2)[:600])


if __name__ == "__main__":
    main()
