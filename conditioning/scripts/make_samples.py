#!/usr/bin/env python
"""Generate clips from a DiT checkpoint and write them out for viewing.

Produces, per conditioning depth k:

* ``sheet_k<k>.png``  -- a contact sheet. Two rows per clip: ground truth on
  top, generation below. The context frames the model was handed are outlined,
  and are identical in both rows by construction, so only what follows the
  outline is the model's own work.
* ``clip<i>_k<k>.gif`` -- ground truth beside generation, animated at 10 Hz,
  the rate the clips were recorded at.

Clips are chosen to span the motion range rather than taken from the head of
the split. The calibration run found 11% of the test split never exceeds
0.1 m/s and the median clip turns 0.27 degrees end to end, so eight consecutive
clips would mostly show the same near-stationary straight drive and say
nothing about how the model handles motion.

Usage:
    python conditioning/scripts/make_samples.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --ckpt-dir <proj>/checkpoints --dit-run dit10m32b100e \\
        --dataset-dir <proj>/data/waymo/driving_wp_f16/test --out-dir <proj>/out/samples
"""
import sys
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

import imageio.v2 as imageio
import numpy as np
import torch

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import encode_latents, stack_latent_stats
from f_toy.engine.dit_sample import sample
from f_toy.evaluation.checkpoints import load_dit, load_vae

from conditioning.lib.dit_eval import condition_raw_frame_cutoff

SCALE = 3          # 96 px is small to judge by eye
PAD = 2
BORDER = (255, 96, 0)


def pick_clips(ann, n):
    """Clips spanning the motion range: stationary, slow, cruising, fast, and
    the sharpest turns present. Returns indices with their reason."""
    pos = ann["ego_position"]
    hdg = ann["ego_heading"]
    step = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    length = step.sum(axis=1)
    turn = np.abs((hdg[:, -1] - hdg[:, 0] + np.pi) % (2 * np.pi) - np.pi)

    picks = []
    picks.append((int(np.argmin(length)), "stationary"))
    for q, name in ((25, "slow"), (50, "median"), (75, "brisk"), (97, "fast")):
        target = np.percentile(length, q)
        picks.append((int(np.argmin(np.abs(length - target))), name))
    order = np.argsort(-turn)
    for j in order[:3]:
        picks.append((int(j), "turning"))

    seen, out = set(), []
    for i, why in picks:
        if i not in seen:
            seen.add(i)
            out.append((i, why, float(length[i]), float(np.degrees(turn[i]))))
        if len(out) >= n:
            break
    return out


def to_uint8(v):
    """(3,T,H,W) float [0,1] -> (T,H,W,3) uint8, upscaled by SCALE."""
    a = (v.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return a.repeat(SCALE, axis=1).repeat(SCALE, axis=2)


def outline(frame, colour=BORDER, w=2):
    f = frame.copy()
    f[:w, :], f[-w:, :], f[:, :w], f[:, -w:] = colour, colour, colour, colour
    return f


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--dit-run", required=True)
    parser.add_argument("--dit-which", choices=["best", "last"], default="best")
    parser.add_argument("--vae-run", default=None)
    parser.add_argument("--vae-which", choices=["best", "last"], default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--ks", default="1,2", help="conditioning depths, comma separated")
    args = parser.parse_args()
    if not args.dataset_dir:
        parser.error("--dataset-dir is required")

    device = args.device or get_device()
    seed = 0 if args.seed is None else args.seed
    ckpt_dir = Path(args.ckpt_dir) / (args.dataset or "cubetoy")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dit, dit_cfg = load_dit(ckpt_dir / args.dit_run, args.dit_which, device)
    vae_run = args.vae_run or dit_cfg["VAE_RUN"]
    vae_which = args.vae_which or dit_cfg.get("VAE_WHICH", "best")
    vae, _ = load_vae(ckpt_dir / vae_run, vae_which, device)
    latent_stats = torch.load(ckpt_dir / vae_run / f"{vae_which}.pt", map_location="cpu",
                              weights_only=False)["latent_stats"]

    plugin = get_dataset_plugin(dit_cfg)
    ds = plugin.DiskDataset(args.dataset_dir)
    ann = np.load(Path(args.dataset_dir) / "annotations.npz")
    chosen = pick_clips(ann, args.num_clips)
    print(f"{len(ds)} clips available; showing {len(chosen)}:")
    for i, why, L, deg in chosen:
        print(f"  clip {i:>5}  {why:<11} path {L:6.2f} m   turn {deg:6.2f} deg")

    videos = torch.stack([ds[i][0] for i, *_ in chosen]).to(device)
    Tz = vae.latent_num_frames(dit_cfg["FRAMES"])
    Hz = Wz = dit_cfg["IMG"] // vae.spatial_compression_factor
    mean, std = stack_latent_stats(latent_stats, Tz, device)

    for k in [int(x) for x in args.ks.split(",")]:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        cutoff = condition_raw_frame_cutoff(k, vae.temporal_compression_factor)
        cond = encode_latents(vae, videos, mean, std)
        shape_z = (len(videos), dit_cfg["VAE_ZDIM"], Tz, Hz, Wz)
        pred = sample(dit, vae, latent_stats, shape_z, num_steps=args.steps, device=device,
                      condition_latent=cond, num_condition_frames=k,
                      target_frames=dit_cfg["FRAMES"])
        print(f"k={k}: context is raw frames 0..{cutoff - 1}, generated {cutoff}..{dit_cfg['FRAMES'] - 1}")

        rows = []
        for n, (idx, why, L, deg) in enumerate(chosen):
            gt, gen = to_uint8(videos[n]), to_uint8(pred[n])
            for strip in (gt, gen):
                marked = [outline(f) if t < cutoff else f for t, f in enumerate(strip)]
                rows.append(np.concatenate(
                    [np.pad(f, ((0, 0), (PAD, PAD), (0, 0)), constant_values=255) for f in marked],
                    axis=1))
            # gt beside generation, animated
            pair = np.concatenate([gt, np.full((gt.shape[0], gt.shape[1], 4, 3), 255, np.uint8), gen],
                                  axis=2)
            imageio.mimsave(out_dir / f"clip{idx}_{why}_k{k}.gif",
                            list(pair), duration=0.1, loop=0)

        sheet = np.concatenate(
            [np.pad(r, ((PAD, PAD), (0, 0), (0, 0)), constant_values=255) for r in rows], axis=0)
        imageio.imwrite(out_dir / f"sheet_k{k}.png", sheet)
        print(f"  wrote sheet_k{k}.png  ({sheet.shape[1]}x{sheet.shape[0]}) and "
              f"{len(chosen)} gifs")

    print(f"\nall output in {out_dir}")
    print("contact sheets: rows alternate ground truth, then generation, per clip.")
    print("outlined frames are the ground-truth context, identical in both rows by construction.")


if __name__ == "__main__":
    main()
