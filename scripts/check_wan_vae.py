#!/usr/bin/env python
"""Zero-shot reconstruction of the released Wan2.1 VAE on womd_7hz_f33 clips.

Three reconstructions of the same val clips, scored against the real frames:

miniwan   vae_50e24b, the stage-1 VAE (trained on 10 Hz, 16-frame clips)
wan96     Wan2.1 VAE at 96 px, the drop-in candidate
wan192    Wan2.1 VAE on a 2x bicubic upsample, decoded and downsampled back
          to 96; not usable as is (24x24 latent grid), it isolates how much
          of any shortfall is resolution

The released Wan2.1 VAE is WanVAE_(dim=96, z_dim=16, dim_mult=[1,2,4,4],
num_res_blocks=2, temperal_downsample=[False,True,True]) on [-1, 1] input,
which f_toy.models.vae vendors verbatim.

    python scripts/check_wan_vae.py --num-clips 500 --out-dir out/wan_vae_zeroshot
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "third_party/Task_specific_JDM")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from f_toy.evaluation.checkpoints import load_vae
from f_toy.models.vae import WanVAE_

PROJ = Path(__file__).resolve().parents[1]
WAN_LOCAL = PROJ / "checkpoints/wan21_vae/Wan2.1_VAE.pth"
WAN_REMOTE = "assets/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def fetch_wan():
    if WAN_LOCAL.exists():
        return
    from vessl.storage.file import VolumeFile
    from cond_eval.unified.volume import open_volume
    WAN_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    vol = open_volume()
    size = [f.size for f in vol.list("assets/Wan2.1-T2V-1.3B") if f.path.endswith("Wan2.1_VAE.pth")][0]
    vol.download_file(VolumeFile(path=WAN_REMOTE, size=size), str(WAN_LOCAL))
    print(f"fetched {WAN_LOCAL} ({WAN_LOCAL.stat().st_size / 1e6:.0f} MB)", flush=True)


def build_wan(dev):
    m = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                temperal_downsample=[False, True, True], dropout=0.0)
    sd = torch.load(WAN_LOCAL, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Wan2.1 VAE load: {len(missing)} missing, {len(unexpected)} unexpected; "
                           f"first {missing[:3]} {unexpected[:3]}")
    n = sum(p.numel() for p in m.parameters())
    print(f"Wan2.1 VAE loaded cleanly, {n / 1e6:.1f}M params", flush=True)
    return m.to(dev).eval()


@torch.no_grad()
def wan_recon(m, x, scale=1):
    """x (B,3,T,H,W) in [0,1] -> recon in [0,1] at the input size."""
    B, C, T, H, W = x.shape
    if scale != 1:
        y = F.interpolate(x.transpose(1, 2).reshape(B * T, C, H, W), scale_factor=scale,
                          mode="bicubic", align_corners=False).clamp(0, 1)
        x_in = y.reshape(B, T, C, H * scale, W * scale).transpose(1, 2)
    else:
        x_in = x
    mu, _ = m.encode(x_in * 2 - 1)
    out = ((m.decode(mu) + 1) / 2)[:, :, :T].clamp(0, 1)
    if scale != 1:
        out = F.interpolate(out.transpose(1, 2).reshape(B * T, C, H * scale, W * scale),
                            size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
        out = out.reshape(B, T, C, H, W).transpose(1, 2)
    return out, tuple(mu.shape)


def scores(pred, gt, lp):
    """Per-clip PSNR, per-frame LPIPS (B,T) and per-frame PSNR (B,T)."""
    B, C, T, H, W = gt.shape
    mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3, 4)).clamp_min(1e-10)
    psnr = (10 * torch.log10(1.0 / mse)).cpu().numpy()
    a = pred.transpose(1, 2).reshape(B * T, C, H, W)
    b = gt.transpose(1, 2).reshape(B * T, C, H, W)
    l = lp(a, b, normalize=True).reshape(B, T).cpu().numpy()
    fmse = ((pred - gt) ** 2).mean(dim=(1, 3, 4)).clamp_min(1e-10)
    fpsnr = (10 * torch.log10(1.0 / fmse)).cpu().numpy()
    return psnr, l, fpsnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="out/wan_vae_zeroshot")
    args = ap.parse_args()
    dev = "cuda"
    out = PROJ / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    D = PROJ / f"data/womd_7hz_f33/{args.split}"
    vids = np.load(D / "videos.npy", mmap_mode="r")
    ann = np.load(D / "annotations.npz")
    stride = max(1, len(vids) // args.num_clips)
    idx = np.arange(0, stride * args.num_clips, stride)[:args.num_clips]
    slice_key = next((k for k in ann.files if k.lower() in ("slice", "slices", "slice_name", "slice_id")), None)
    print(f"{args.split}: {len(vids)} clips, scoring {len(idx)} (every {stride}th); "
          f"clip shape {vids.shape[1:]}; slice key {slice_key}", flush=True)

    fetch_wan()
    wan = build_wan(dev)
    mini, _ = load_vae(PROJ / "checkpoints/waymo/vae_50e24b", "best", dev)
    mini.eval()
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex", verbose=False).to(dev).eval()

    names = ("miniwan", "wan96", "wan192")
    acc = {n: {"psnr": [], "lpips": [], "fl": [], "fp": []} for n in names}
    lat = {}
    keep = {}
    t0 = time.time()
    for s in range(0, len(idx), args.batch):
        bi = idx[s:s + args.batch]
        x = torch.from_numpy(np.stack([vids[i] for i in bi])).to(dev).float() / 255.0
        with torch.no_grad():
            mu, _ = mini.encode(x)
            rec_m = mini.decode(mu)[:, :, :x.shape[2]].clamp(0, 1)
            lat["miniwan"] = tuple(mu.shape)
            rec_w, lat["wan96"] = wan_recon(wan, x, 1)
            rec_w2, lat["wan192"] = wan_recon(wan, x, 2)
            for n, r in zip(names, (rec_m, rec_w, rec_w2)):
                p, l, fp = scores(r.float(), x, lp)
                acc[n]["psnr"].append(p)
                acc[n]["lpips"].append(l.mean(1))
                acc[n]["fl"].append(l)
                acc[n]["fp"].append(fp)
        if s == 0:
            keep = dict(gt=x[:4].cpu(), miniwan=rec_m[:4].cpu(), wan96=rec_w[:4].cpu(), wan192=rec_w2[:4].cpu())
        if (s // args.batch) % 10 == 0:
            print(f"  {s + len(bi)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)

    res = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in acc.items()}
    summary = {}
    print("")
    print(f"{'':10s} {'PSNR':>8s} {'LPIPS':>8s}   latent (B,C,T,H,W)")
    for n in names:
        summary[n] = dict(psnr=float(res[n]["psnr"].mean()), lpips=float(res[n]["lpips"].mean()),
                          lpips_per_frame=[round(float(v), 4) for v in res[n]["fl"].mean(0)],
                          psnr_per_frame=[round(float(v), 3) for v in res[n]["fp"].mean(0)],
                          latent_shape=list(lat[n]))
        print(f"{n:10s} {summary[n]['psnr']:8.3f} {summary[n]['lpips']:8.4f}   {lat[n]}")
    if slice_key:
        sl = ann[slice_key][idx]
        summary["by_slice"] = {}
        print("")
        print("LPIPS by slice")
        for v in np.unique(sl):
            m = sl == v
            row = {n: float(res[n]["lpips"][m].mean()) for n in names}
            summary["by_slice"][str(v)] = dict(n=int(m.sum()), **row)
            print(f"  {str(v):14s} n={int(m.sum()):4d}  " + "  ".join(f"{n} {row[n]:.4f}" for n in names))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    col = {"miniwan": "#2a78d6", "wan96": "#eb6834", "wan192": "#1baf7a"}
    lab = {"miniwan": "MiniWan vae_50e24b (stage 1)", "wan96": "Wan2.1 VAE at 96 px",
           "wan192": "Wan2.1 VAE at 192 px, back to 96"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for n in names:
        ax.plot(res[n]["fl"].mean(0), color=col[n], lw=2, marker="o", ms=3, label=lab[n])
    ax.set_xlabel("frame index (7 Hz)", color="#55555a")
    ax.set_title(f"Reconstruction LPIPS per frame, {len(idx)} {args.split} clips, lower is better",
                 loc="left", fontsize=11)
    ax.grid(True, color="#dcdcd8", lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "lpips_per_frame.png", dpi=170, facecolor="white")

    frames = [0, 8, 16, 24, 32]
    rows = ["gt", "miniwan", "wan96", "wan192"]
    S = 3
    tile = 96 * S
    for c in range(keep["gt"].shape[0]):
        canvas = np.full((len(rows) * (tile + 4), len(frames) * (tile + 4), 3), 255, np.uint8)
        for r, key in enumerate(rows):
            for j, f in enumerate(frames):
                im = (keep[key][c, :, f].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                im = im.repeat(S, 0).repeat(S, 1)
                canvas[r * (tile + 4):r * (tile + 4) + tile, j * (tile + 4):j * (tile + 4) + tile] = im
        imageio.imwrite(out / f"sheet_clip{int(idx[c])}.png", canvas)
    (out / "README.txt").write_text(
        "sheet_clip*.png rows: ground truth, MiniWan vae_50e24b, Wan2.1 at 96 px, Wan2.1 at 192 px. "
        "Columns: frames 0, 8, 16, 24, 32 (7 Hz). summary.json has means, per-frame curves and slices.")
    print("wrote", out)


if __name__ == "__main__":
    main()
