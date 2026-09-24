#!/usr/bin/env python
"""Train the ego-motion probe, and measure the floors that decide whether its
output can be read as a trajectory metric.

The floors matter as much as the training. Reported at the end:

* **real held-out** ADE/FDE/heading. Any difference between conditions smaller
  than this is unreadable, so it caps what the whole trajectory axis resolves.
* **VAE round-trip** ADE/FDE/heading, on reconstructions of the same clips. The
  ego motion is identical, so all the extra error is attributable to appearance
  degradation alone. If it sits far above the real floor, the probe is
  appearance-sensitive and the trajectory metric is partly a blur detector.
  ``--vae-aug`` trains through that degradation, which is the fix.
* **two sanity controls**: a clip repeating one frame 16 times must read as
  near-zero motion, and a time-reversed clip must read as reversed motion.
  Either failing means the probe learned scene priors rather than motion.

Usage:
    python scripts/train_probe.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --ckpt-dir <proj>/checkpoints \\
        --dataset-dir <proj>/data/driving_wp_f16/train \\
        --val-dataset-dir <proj>/data/driving_wp_f16/test \\
        --out-dir <proj>/out/probe --epochs 20
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return jdm


JDM = _bootstrap()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset_plugin

from cond_eval.pose_probe import PoseProbe, increments_from_labels, integrate_increments

XY_SCALE = 10.0   # about the mean path length in metres, so the two loss terms are comparable


def torch_integrate(inc):
    """Differentiable counterpart of ``integrate_increments``, origin start.
    Returns (B,T-1,2), excluding the origin."""
    dth = inc[..., 2]
    th = torch.cumsum(dth, dim=1)
    th = torch.cat([torch.zeros_like(th[:, :1]), th[:, :-1]], dim=1)
    c, s = torch.cos(th), torch.sin(th)
    fwd, lat = inc[..., 0], inc[..., 1]
    step = torch.stack([c * fwd - s * lat, s * fwd + c * lat], dim=-1)
    return torch.cumsum(step, dim=1)


def pose_errors(pred_inc, tgt_inc):
    """ADE, FDE (m) and mean heading error (deg), per sample, from increments."""
    pxy, ph = integrate_increments(pred_inc)
    txy, th = integrate_increments(tgt_inc)
    ade = np.linalg.norm(pxy - txy, axis=-1).mean(axis=-1)
    fde = np.linalg.norm(pxy[:, -1] - txy[:, -1], axis=-1)
    d = (ph - th + np.pi) % (2 * np.pi) - np.pi
    return ade, fde, np.degrees(np.abs(d)).mean(axis=-1)


def report(name, ade, fde, hdg):
    print(f"  {name:26s} ADE {np.mean(ade):7.4f} m   FDE {np.mean(fde):7.4f} m   "
          f"heading {np.mean(hdg):6.3f} deg   (p95 ADE {np.percentile(ade, 95):.4f})")
    return dict(ade=float(np.mean(ade)), fde=float(np.mean(fde)),
                heading_deg=float(np.mean(hdg)), ade_p95=float(np.percentile(ade, 95)))


@torch.no_grad()
def evaluate(model, loader, targets, device, vae=None):
    model.eval()
    preds, tgts, n = [], [], 0
    for videos, _ in loader:
        videos = videos.to(device)
        if vae is not None:
            videos = vae(videos)[0].clamp(0, 1)
        preds.append(model.predict_increments(videos))
        tgts.append(targets[n:n + len(videos)])
        n += len(videos)
    return pose_errors(np.concatenate(preds), np.concatenate(tgts))


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="tail fraction of the TRAIN split held out for per-epoch "
                             "checks; the test split is never trained on")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--xy-weight", type=float, default=0.2,
                        help="weight on the integrated-track term, so integration error "
                             "is penalised directly and not only per-step error")
    parser.add_argument("--vae-aug", type=float, default=0.0,
                        help="probability of replacing a training clip with its VAE "
                             "round-trip; the fix if the round-trip floor is poor")
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not (args.dataset_dir and args.val_dataset_dir):
        parser.error("--dataset-dir (train) and --val-dataset-dir (test) are both required")

    device = args.device or get_device()
    epochs = args.epochs or 20
    batch = args.batch or 64
    seed = 0 if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"DATASET": args.dataset or "waymo"}
    plugin = get_dataset_plugin(cfg)
    train_full = plugin.DiskDataset(args.dataset_dir)
    test_ds = plugin.DiskDataset(args.val_dataset_dir)
    print(f"train split {len(train_full)} clips, test split {len(test_ds)} clips")

    def split_targets(root):
        ann = np.load(Path(root) / "annotations.npz")
        return increments_from_labels(ann["ego_position"], ann["ego_heading"])

    tgt_train_all = split_targets(args.dataset_dir)
    tgt_test = split_targets(args.val_dataset_dir)

    n_val = int(len(train_full) * args.val_frac)
    fit_idx = np.arange(len(train_full) - n_val)
    val_idx = np.arange(len(train_full) - n_val, len(train_full))
    fit_ds, val_ds = Subset(train_full, fit_idx), Subset(train_full, val_idx)
    tgt_fit, tgt_val = tgt_train_all[fit_idx], tgt_train_all[val_idx]
    print(f"fit {len(fit_ds)} / held-out-from-train {len(val_ds)} / test {len(test_ds)}")

    ps_path = Path(args.dataset_dir) / "pixel_stats.json"
    pixel_mean = pixel_std = None
    if ps_path.exists():
        ps = json.loads(ps_path.read_text())
        pixel_mean, pixel_std = ps.get("mean"), ps.get("std")
        print(f"pixel stats: mean {pixel_mean}, std {pixel_std}")

    inc_mean = tgt_fit.reshape(-1, 3).mean(0)
    inc_std = tgt_fit.reshape(-1, 3).std(0)
    print(f"increment mean {inc_mean}  std {inc_std}")

    model = PoseProbe(width=args.width, pixel_mean=pixel_mean, pixel_std=pixel_std).to(device)
    model.inc_mean.copy_(torch.as_tensor(inc_mean))
    model.inc_std.copy_(torch.as_tensor(inc_std))
    print(f"PoseProbe {sum(p.numel() for p in model.parameters()):,} params")

    from f_toy.evaluation.checkpoints import load_vae
    vae_dir = Path(args.ckpt_dir) / (args.dataset or "waymo") / args.vae_run
    vae = None
    if args.vae_aug > 0:
        vae, _ = load_vae(vae_dir, args.vae_which, device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        print(f"VAE round-trip augmentation at p={args.vae_aug}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(fit_ds, batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=args.workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=epochs * len(loader), pct_start=0.1)

    m = torch.as_tensor(inc_mean, device=device)
    sd = torch.as_tensor(inc_std, device=device)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            tgt = torch.as_tensor(
                increments_from_labels(labels["ego_position"].numpy(),
                                       labels["ego_heading"].numpy()), device=device)
            if vae is not None and np.random.rand() < args.vae_aug:
                with torch.no_grad():
                    videos = vae(videos)[0].clamp(0, 1)

            pred_n = model(videos)
            loss_inc = F.l1_loss(pred_n, (tgt - m) / sd)
            loss_xy = F.l1_loss(torch_integrate(pred_n * sd + m),
                                torch_integrate(tgt)) / XY_SCALE
            loss = loss_inc + args.xy_weight * loss_xy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(videos)
            n += len(videos)

        ade, fde, hdg = evaluate(model, val_loader, tgt_val, device)
        print(f"[probe] ep {ep:>3}/{epochs} train_loss {tot / n:.5f}  "
              f"val ADE {np.mean(ade):.4f} m  FDE {np.mean(fde):.4f} m  "
              f"heading {np.mean(hdg):.3f} deg  ({time.time() - t0:.0f}s)", flush=True)

    ckpt = out_dir / "probe.pt"
    torch.save(dict(model=model.state_dict(), width=args.width, seed=seed,
                    inc_mean=inc_mean, inc_std=inc_std, epochs=epochs,
                    train_dir=str(args.dataset_dir)), ckpt)
    print(f"saved {ckpt}")

    print("\n=== floors, on the held-out TEST split ===")
    summary = {}
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=args.workers)
    summary["floor_real"] = report("real frames", *evaluate(model, test_loader, tgt_test, device))

    vae_eval, _ = load_vae(vae_dir, args.vae_which, device)
    vae_eval.eval()
    summary["floor_vae_roundtrip"] = report(
        "VAE round-trip", *evaluate(model, test_loader, tgt_test, device, vae=vae_eval))

    print("\n=== sanity controls ===")
    videos, labels = next(iter(test_loader))
    videos = videos.to(device)
    frozen = videos[:, :, :1].repeat(1, 1, videos.shape[2], 1, 1)
    xy_f, _ = integrate_increments(model.predict_increments(frozen))
    disp = np.linalg.norm(xy_f[:, -1], axis=-1)
    print(f"  repeated single frame  net displacement {disp.mean():.4f} m "
          f"(max {disp.max():.4f}) -- should be about 0")
    summary["control_static_displacement_m"] = float(disp.mean())

    tgt_b = increments_from_labels(labels["ego_position"].numpy(), labels["ego_heading"].numpy())
    fwd_true = tgt_b[..., 0].sum(axis=1)
    fwd_rev = model.predict_increments(torch.flip(videos, dims=[2]))[..., 0].sum(axis=1)
    print(f"  time-reversed          forward sum {fwd_rev.mean():+.4f} m against "
          f"{fwd_true.mean():+.4f} m forwards -- sign should flip")
    summary["control_reversed_forward_m"] = float(fwd_rev.mean())
    summary["control_forward_m"] = float(fwd_true.mean())

    (out_dir / "probe_floors.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir / 'probe_floors.json'}")


if __name__ == "__main__":
    main()
