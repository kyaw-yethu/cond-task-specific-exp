#!/usr/bin/env python
"""Show the VAE round-trip frames the `--vae-aug` probe trains on.

Three rows per clip: the real frames, the VAE reconstruction the augmentation
substitutes, and their absolute difference amplified so the loss is visible. The
difference row is the point: it shows *what* the probe is being asked to read
motion from once appearance degrades, which is why the round-trip floor sat at
0.4351 m against 0.2105 m on real frames.

Also prints per-clip PSNR so the visual impression has a number attached.

Usage:
    python conditioning/scripts/show_vae_recon.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --ckpt-dir <proj>/checkpoints --dataset-dir <proj>/data/driving_wp_f16/test \\
        --out-dir <proj>/out/vae_recon
"""
import sys
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

import imageio.v2 as imageio
import numpy as np
import torch

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset_plugin
from f_toy.evaluation.checkpoints import load_vae

SCALE, PAD, AMP = 3, 2, 4.0


def to_uint8(v):
    a = (v.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return a.repeat(SCALE, axis=1).repeat(SCALE, axis=2)


def strip(frames, label_band=0):
    return np.concatenate(
        [np.pad(f, ((0, 0), (PAD, PAD), (0, 0)), constant_values=255) for f in frames], axis=1)


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--num-clips", type=int, default=5)
    args = parser.parse_args()

    device = args.device or get_device()
    ckpt_dir = Path(args.ckpt_dir) / (args.dataset or "waymo")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, _ = load_vae(ckpt_dir / args.vae_run, args.vae_which, device)
    vae.eval()

    plugin = get_dataset_plugin({"DATASET": args.dataset or "waymo"})
    ds = plugin.DiskDataset(args.dataset_dir)
    ann = np.load(Path(args.dataset_dir) / "annotations.npz")
    pos = ann["ego_position"]
    length = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1)

    # Span the motion range: the probe has to work on stationary and fast clips alike.
    picks = []
    for q, name in ((1, "stationary"), (35, "slow"), (60, "median"), (85, "brisk"), (99, "fast")):
        i = int(np.argmin(np.abs(length - np.percentile(length, q))))
        if i not in [p[0] for p in picks]:
            picks.append((i, name, float(length[i])))
    picks = picks[:args.num_clips]

    videos = torch.stack([ds[i][0] for i, *_ in picks]).to(device)
    with torch.no_grad():
        recon = vae(videos)[0].clamp(0, 1)

    rows, crops = [], {}
    print(f"{'clip':>6} {'kind':<11} {'path m':>7} {'PSNR dB':>8}")
    for n, (idx, name, L) in enumerate(picks):
        mse = torch.mean((videos[n] - recon[n]) ** 2).item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"{idx:>6} {name:<11} {L:>7.2f} {psnr:>8.2f}")

        real, rec = to_uint8(videos[n]), to_uint8(recon[n])
        diff = np.clip(np.abs(real.astype(np.int16) - rec.astype(np.int16)) * AMP,
                       0, 255).astype(np.uint8)
        for band in (real, rec, diff):
            rows.append(strip(band))
        rows.append(np.full((6, rows[-1].shape[1], 3), 255, np.uint8))  # gap between clips

        pair = np.concatenate(
            [real, np.full((real.shape[0], real.shape[1], 4, 3), 255, np.uint8), rec], axis=2)
        imageio.mimsave(out_dir / f"clip{idx}_{name}_real_vs_vae.gif",
                        list(pair), duration=0.1, loop=0)

        # A readable standalone crop: the three rows for this clip, downscaled.
        # The full sheet is 4672 px wide and renders uselessly in a document.
        block = np.concatenate([strip(b) for b in (real, rec, diff)], axis=0)
        crops[name] = block[::SCALE, ::SCALE]
        imageio.imwrite(out_dir / f"view_{name}.png", crops[name])

    sheet = np.concatenate(
        [np.pad(r, ((PAD, PAD), (0, 0), (0, 0)), constant_values=255) for r in rows], axis=0)
    imageio.imwrite(out_dir / "vae_recon_sheet.png", sheet)
    print(f"\nwrote {out_dir / 'vae_recon_sheet.png'} ({sheet.shape[1]}x{sheet.shape[0]})")
    for name in crops:
        print(f"wrote {out_dir / f'view_{name}.png'} "
              f"({crops[name].shape[1]}x{crops[name].shape[0]})")
    print("rows per clip: real, VAE reconstruction, |difference| amplified "
          f"{AMP:g}x; GIFs are real beside reconstruction")


if __name__ == "__main__":
    main()
