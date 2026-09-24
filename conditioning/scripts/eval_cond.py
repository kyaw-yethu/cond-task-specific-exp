#!/usr/bin/env python
"""Evaluate one condition of the stage-1 screen: both axes, on all 4,000 test clips.

Loads a `conditioning/scripts/train_dit_cond.py` checkpoint, which `f_toy`'s `load_dit` cannot
read because the payload carries the conditioner and the cond metadata alongside
the DiT. `f_toy.engine.dit_sample.sample` already accepts `crossattn_emb`, so
conditioned sampling needs no wrapper: the r tokens go straight in.

**r is recomputed here rather than read from the training cache.** The cache
would align by row index, and `run_dit_eval` iterates the loader without passing
indices, so reading it would rest on an unchecked assumption. Recomputing costs
about 75 s over the whole split at fixed k=2, which is not worth the risk.

k and the encoder window are read from the checkpoint, where training recorded
them from its cache. Checkpoints from before that was recorded are the k=2 runs,
so the fallbacks are k=2 and a 6-frame window.

Both axes come out together: PSNR/SSIM/LPIPS over the generated frame range,
FVD16 over the whole clip, and ADE/FDE/DTW/heading plus DrivingGen's
reference-free composite from the ego-motion probe, with per-clip values in
`per_clip.npz` for the paired bootstraps.

Usage:
    python conditioning/scripts/eval_cond.py --jdm third_party/Task_specific_JDM --dataset waymo \\
        --tag V --ckpt-dir <proj>/checkpoints \\
        --dataset-dir <proj>/data/driving_wp_f16/test \\
        --probe <proj>/out/probe_vaeaug/probe.pt \\
        --eval-dir <proj>/out/eval_V
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
        sys.exit(f"no f_toy under {jdm}; pass --jdm")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    return jdm


JDM = _bootstrap()

import torch
from torch.utils.data import DataLoader, Subset

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset_plugin
from f_toy.engine.dit import build_dit
from f_toy.engine.dit_sample import sample
from f_toy.evaluation.checkpoints import load_vae

from conditioning.lib.cond_dit import ConditionedDiT, CrossAttnConditioner
from conditioning.lib.dit_eval import condition_raw_frame_cutoff, run_dit_eval
from conditioning.lib.pose_probe import build_pose_fn
from conditioning.lib.r_sources import build_cond_source



def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM)
    parser.add_argument("--tag", required=True, help="checkpoints/<dataset>/<tag>/")
    parser.add_argument("--which", choices=["best", "last"], default="best")
    parser.add_argument("--probe", default=None,
                        help="probe .pt for the trajectory axis; omit for visual only")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-fvd", action="store_true")
    parser.add_argument("--vae-run", default="vae_50e24b")
    parser.add_argument("--vae-which", default="best")
    parser.add_argument("--i3d-path", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not args.dataset_dir:
        parser.error("--dataset-dir is required (the held-out test split)")

    proj = Path(__file__).resolve().parents[2]
    device = args.device or get_device()
    ckpt_dir = Path(args.ckpt_dir) / (args.dataset or "waymo")
    run_dir = ckpt_dir / args.tag

    ck = torch.load(run_dir / f"{args.which}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cond, cond_run = ck.get("cond", "none"), ck.get("cond_run")
    fixed_k = int(cfg.get("DIT_FIXED_K") or 2)
    width = int(cfg.get("COND_CONTEXT_WIDTH") or 6)
    print(f"{args.tag}/{args.which}: epoch {ck.get('epoch')}, "
          f"val_flow {ck.get('val_flow'):.6f}, cond={cond} ({cond_run}), "
          f"fixed_k={fixed_k}, context width {width}")

    # ---- model ----
    core = build_dit(cfg).to(device)
    core.load_state_dict(ck["model"])
    core.eval()
    conditioner = None
    if ck.get("conditioner") is not None:
        # Shapes come from the conditioner's own positional embedding, (1, N, d),
        # rather than the payload's cond_dim/cond_grid: in the cached training
        # path the encoder is never built, so those fields are None.
        n_tok, dim = ck["conditioner"]["pos"].shape[1:]
        conditioner = CrossAttnConditioner(int(dim), int(n_tok)).to(device)
        conditioner.load_state_dict(ck["conditioner"])
        conditioner.eval()
        print(f"conditioner: {n_tok} tokens of width {dim}")
    model = ConditionedDiT(core, conditioner).to(device).eval()

    vae, _ = load_vae(ckpt_dir / args.vae_run, args.vae_which, device)
    vae.eval()
    latent_stats = torch.load(ckpt_dir / args.vae_run / f"{args.vae_which}.pt",
                              map_location="cpu", weights_only=False)["latent_stats"]

    source = None
    if cond != "none":
        source = build_cond_source(cond, ckpt_dir, device,
                                   context_width=width,
                                   img_size=(96 if cond == "traj_vae" else 256),
                                   run=cond_run, which="last")
        print(f"conditioning source: {source.num_tokens} tokens of width {source.dim}")
        if (source.num_tokens, source.dim) != (int(n_tok), int(dim)):
            raise RuntimeError(f"source gives {source.num_tokens} x {source.dim}, the "
                               f"conditioner was trained on {n_tok} x {dim}")

    dataset = get_dataset_plugin(cfg).DiskDataset(args.dataset_dir)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))
    loader = DataLoader(dataset, batch_size=args.batch or 16, shuffle=False, num_workers=4)
    print(f"test set: {len(dataset)} clips")

    cutoff = condition_raw_frame_cutoff(fixed_k, vae.temporal_compression_factor)
    cut_t = torch.full((args.batch or 16,), cutoff, device=device, dtype=torch.long)

    def sample_fn(videos, shape_z, condition_latent, num_cond, target_frames):
        ctx = None
        if source is not None:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                r = source.encode(videos, cut_t[:videos.shape[0]])
            ctx = conditioner(r.float())
        return sample(model.dit, vae, latent_stats, shape_z, num_steps=args.steps,
                      device=device, crossattn_emb=ctx,
                      condition_latent=condition_latent,
                      num_condition_frames=num_cond, target_frames=target_frames)

    pose_fn = build_pose_fn(device, args.probe) if args.probe else None
    if pose_fn:
        print(f"probe: {args.probe}")

    eval_dir = Path(args.eval_dir) if args.eval_dir else proj / "out" / f"eval_{args.tag}"
    run_dit_eval(sample_fn, vae, latent_stats, loader, cfg, device, eval_dir,
                 num_condition_frames=fixed_k, steps=args.steps,
                 run=args.tag, which=args.which, dataset_dir=args.dataset_dir,
                 log_every=args.log_every, fvd=not args.no_fvd,
                 i3d_path=args.i3d_path or str(proj / ".runtime/i3d/i3d_torchscript.pt"),
                 pose_fn=pose_fn, pose_floor=False,
                 extra=dict(cond=cond, cond_run=cond_run or "", epoch=ck.get("epoch", -1),
                            fixed_k=fixed_k, context_width=width,
                            cond_tokens=(int(n_tok) if conditioner else 0),
                            cond_dim=(int(dim) if conditioner else 0),
                            dit_width=int(cfg["DIT_MODEL_CHANNELS"]),
                            dit_blocks=int(cfg["DIT_NUM_BLOCKS"]),
                            dit_params=sum(p.numel() for p in core.parameters())))


if __name__ == "__main__":
    main()
