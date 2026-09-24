#!/usr/bin/env python
"""Qualitative comparison: ground truth beside every condition, same clips.

For each selected clip, generates from all five conditions with the same seed
and the same context frames, then tiles GT | N | T | V | D | G into one
animated GIF and one still contact sheet.

Clips are chosen by path-length percentile so the set spans stationary to fast,
and the turning clips are included separately, since turning is where the
baseline visibly fails and so where conditioning has the most room to show.

Usage:
    python scripts/make_comparison.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --ckpt-dir <proj>/checkpoints --dataset-dir <proj>/data/driving_wp_f16/test \\
        --tags N T V D G --out-dir <proj>/out/comparison
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return jdm


JDM = _bootstrap()

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import build_dit, encode_latents, stack_latent_stats
from f_toy.engine.dit_sample import sample
from f_toy.evaluation.checkpoints import load_vae

from cond_eval.cond_dit import CrossAttnConditioner
from cond_eval.dit_eval import condition_raw_frame_cutoff
from cond_eval.r_sources import build_cond_source

SCALE, PAD = 3, 2
PCT_NAME = {50: "median", 85: "brisk", 99: "fast"}   # the default set keeps its old names
LABEL_H = 16


def to_uint8(v):
    a = (v.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return a.repeat(SCALE, axis=1).repeat(SCALE, axis=2)


def label_band(width, text):
    """A thin white band with the panel name burnt in, so a tiled GIF is
    self-describing without a legend beside it."""
    band = np.full((LABEL_H, width, 3), 255, np.uint8)
    img = Image.fromarray(band)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_H - 6)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 1), text, fill=(20, 20, 20), font=font)
    return np.asarray(img)


def with_label(clip, text):
    """Prepend the label band to every frame of a (T, H, W, 3) clip."""
    band = label_band(clip.shape[2], text)
    return np.concatenate(
        [np.repeat(band[None], clip.shape[0], axis=0), clip], axis=1)


PANEL_TEXT = {
    "GT": "GT  ground truth",
    "N": "N  null context",
    "D": "D  Drive-JEPA domain",
    "T": "T  Drive-JEPA task",
    "V": "V  Traj-VAE",
    "G": "G  V-JEPA2 ViT-H",
}


def panel_text(tag):
    """Label for a run tag; a suffixed tag such as ``G1`` reads as its letter."""
    return PANEL_TEXT.get(tag) or PANEL_TEXT.get(tag[:1], tag)


def grid(panels, ncol=3):
    """Tile labelled (T, H, W, 3) panels into a (T, H*rows, W*ncol, 3) mosaic,
    padding the last row with white so every row is the same width."""
    h, w = panels[0].shape[1:3]
    blank = np.full_like(panels[0], 255)
    padded = list(panels) + [blank] * (-len(panels) % ncol)
    rows = [np.concatenate(padded[i:i + ncol], axis=2)
            for i in range(0, len(padded), ncol)]
    return np.concatenate(rows, axis=1)


def load_condition(tag, ckpt_dir, device, which="best"):
    """Returns ``(dit, conditioner, source, meta)`` for one trained condition."""
    ck = torch.load(Path(ckpt_dir) / tag / f"{which}.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["cfg"]
    dit = build_dit(cfg).to(device)
    dit.load_state_dict(ck["model"])
    dit.eval()

    conditioner = source = None
    cond = ck.get("cond", "none")
    if ck.get("conditioner") is not None:
        n_tok, dim = ck["conditioner"]["pos"].shape[1:]
        conditioner = CrossAttnConditioner(int(dim), int(n_tok)).to(device)
        conditioner.load_state_dict(ck["conditioner"])
        conditioner.eval()
        source = build_cond_source(cond, ckpt_dir, device,
                                   context_width=int(cfg.get("COND_CONTEXT_WIDTH") or 6),
                                   img_size=(96 if cond == "traj_vae" else 256),
                                   run=ck["cond_run"], which="last")
    return dit, conditioner, source, dict(cond=cond, epoch=ck.get("epoch"),
                                          val_flow=ck.get("val_flow"),
                                          fixed_k=int(cfg.get("DIT_FIXED_K") or 2))


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--tags", nargs="+", default=["N", "T", "V", "D", "G"])
    parser.add_argument("--which", default="best")
    parser.add_argument("--num-clips", type=int, default=6)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 85, 99],
                        help="path-length percentiles to pick first; the sharpest "
                             "turns fill the remaining --num-clips slots")
    parser.add_argument("--gif-cols", type=int, default=3,
                        help="panels per row in the GIF mosaic")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--fixed-k", type=int, default=2,
                        help="latent context frames; must match every tag's training k")
    args = parser.parse_args()
    FIXED_K = args.fixed_k

    device = args.device or get_device()
    seed = 0 if args.seed is None else args.seed
    ckpt_dir = Path(args.ckpt_dir) / (args.dataset or "waymo")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, _ = load_vae(ckpt_dir / args.vae_run, args.vae_which, device)
    vae.eval()
    latent_stats = torch.load(ckpt_dir / args.vae_run / f"{args.vae_which}.pt",
                              map_location="cpu", weights_only=False)["latent_stats"]

    # ---- clip selection: span motion, and include the sharpest turns ----
    plugin = get_dataset_plugin({"DATASET": args.dataset or "waymo"})
    ds = plugin.DiskDataset(args.dataset_dir)
    ann = np.load(Path(args.dataset_dir) / "annotations.npz")
    pos, hdg = ann["ego_position"], ann["ego_heading"]
    length = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1)
    turn = np.abs((hdg[:, -1] - hdg[:, 0] + np.pi) % (2 * np.pi) - np.pi)

    picks, seen = [], set()
    for q in args.percentiles:
        i = int(np.argmin(np.abs(length - np.percentile(length, q))))
        if i not in seen:
            seen.add(i); picks.append((i, PCT_NAME.get(q, f"p{q:g}")))
    for j in np.argsort(-turn):
        if len(picks) >= args.num_clips:
            break
        if int(j) not in seen:
            seen.add(int(j)); picks.append((int(j), "turning"))
    picks = picks[:args.num_clips]
    for i, name in picks:
        print(f"clip {i:>5} {name:<9} path {length[i]:6.2f} m  turn "
              f"{np.degrees(turn[i]):6.2f} deg")

    videos = torch.stack([ds[i][0] for i, _ in picks]).to(device)
    B = videos.shape[0]
    Tz = vae.latent_num_frames(videos.shape[2])
    Hz = Wz = videos.shape[-1] // vae.spatial_compression_factor
    mean, std = stack_latent_stats(latent_stats, Tz, device)
    cutoff = condition_raw_frame_cutoff(FIXED_K, vae.temporal_compression_factor)
    cut_t = torch.full((B,), cutoff, device=device, dtype=torch.long)
    cond_latent = encode_latents(vae, videos, mean, std)

    rows = {"GT": to_uint8_batch(videos)}
    order = ["GT"]
    for tag in args.tags:
        p = ckpt_dir / tag / f"{args.which}.pt"
        if not p.exists():
            print(f"skip {tag}: no {p}")
            continue
        dit, conditioner, source, meta = load_condition(tag, ckpt_dir, device, args.which)
        if meta["fixed_k"] != FIXED_K:
            raise SystemExit(f"{tag} was trained at k={meta['fixed_k']}, asked for k={FIXED_K}")
        ctx = None
        if source is not None:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                r = source.encode(videos, cut_t)
            ctx = conditioner(r.float())
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            pred = sample(dit, vae, latent_stats,
                          (B, int(cond_latent.shape[1]), Tz, Hz, Wz),
                          num_steps=args.steps, device=device, crossattn_emb=ctx,
                          condition_latent=cond_latent, num_condition_frames=FIXED_K,
                          target_frames=videos.shape[2])
        rows[tag] = to_uint8_batch(pred)
        order.append(tag)
        print(f"{tag}: sampled (epoch {meta['epoch']}, val_flow {meta['val_flow']:.6f})")
        del dit, conditioner, source
        torch.cuda.empty_cache()

    # ---- write per clip: a labelled mosaic GIF and a still sheet ----
    for n, (idx, name) in enumerate(picks):
        panels = [with_label(rows[t][n], panel_text(t)) for t in order]
        gif = grid(panels, ncol=args.gif_cols)
        imageio.mimsave(out_dir / f"clip{idx}_{name}_GT_vs_{'_'.join(order[1:])}.gif",
                        list(gif), duration=0.12, loop=0)

        sheet_rows = []
        for t in order:
            strip = np.concatenate(
                [np.pad(f, ((0, 0), (PAD, PAD), (0, 0)), constant_values=255)
                 for f in rows[t][n]], axis=1)
            strip = np.pad(strip, ((PAD, PAD), (0, 0), (0, 0)), constant_values=255)
            sheet_rows.append(np.concatenate(
                [label_band(strip.shape[1], panel_text(t)), strip], axis=0))
        imageio.imwrite(out_dir / f"sheet_clip{idx}_{name}.png",
                        np.concatenate(sheet_rows, axis=0))

    (out_dir / "ROW_ORDER.txt").write_text(
        "Every panel carries its own burnt-in label, so this file is a cross-check\n"
        "rather than the only key.\n\n"
        f"GIF: a {min(args.gif_cols, len(order))}-wide mosaic, reading left to right then down:\n"
        + "\n".join(f"  {i + 1}. {panel_text(t)}" for i, t in enumerate(order))
        + "\n\nSheet: the same conditions as rows, top to bottom, all 16 frames "
          "left to right.\n"
        + f"\nContext frames: raw 0..{cutoff - 1} are ground truth in every panel "
          f"(k={FIXED_K}).\nGenerated: raw {cutoff}..{videos.shape[2] - 1}.\n"
          f"Sampling seed {seed}, {args.steps} Euler steps, identical across panels.\n")
    print(f"\nwrote {len(picks)} GIFs and sheets to {out_dir}")
    print("row order:", " | ".join(order))


def to_uint8_batch(v):
    return np.stack([to_uint8(v[i]) for i in range(v.shape[0])])


if __name__ == "__main__":
    main()
