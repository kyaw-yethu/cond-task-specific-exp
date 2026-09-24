#!/usr/bin/env python
"""Evaluate a trained MiniDiT checkpoint with ``cond_eval``'s metrics.

Mirrors ``Task_specific_JDM/scripts/eval_dit.py`` and swaps in
``cond_eval.dit_eval.run_dit_eval``, which adds the trajectory axis on top of
the same PSNR/SSIM/LPIPS/FVD16. Lives here rather than there because that repo
is read-only for this project.

Without ``--pose-module`` it computes the visual metrics only, which is the
whole of what can be measured until the ego-motion probe exists.

Usage:
    python scripts/eval_dit.py --jdm third_party/Task_specific_JDM \\
        --dataset waymo --dit-run dit10m32b100e \\
        --ckpt-dir /root/checkpoints \\
        --dataset-dir /root/data/waymo/driving_wp_f16/test \\
        --max-samples 64 --steps 50            # smoke test
"""
import importlib
import sys
from pathlib import Path


def _bootstrap_paths():
    """``--jdm`` has to be read before f_toy can be imported, so it is parsed
    off argv by hand rather than through the shared parser it provides."""
    jdm = None
    for i, a in enumerate(sys.argv):
        if a == "--jdm" and i + 1 < len(sys.argv):
            jdm = sys.argv[i + 1]
        elif a.startswith("--jdm="):
            jdm = a.split("=", 1)[1]
    if jdm is None:
        for cand in ("third_party/Task_specific_JDM", Path(__file__).resolve().parents[2] / "Task_specific_JDM"):
            if Path(cand, "f_toy").is_dir():
                jdm = str(cand)
                break
    if jdm is None or not Path(jdm, "f_toy").is_dir():
        sys.exit("cannot find Task_specific_JDM; pass --jdm /path/to/Task_specific_JDM")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return jdm


JDM = _bootstrap_paths()
PROJECT = Path(__file__).resolve().parents[1]

import torch
from torch.utils.data import DataLoader, Subset

from f_toy.config import build_arg_parser, get_device
from f_toy.data import get_dataset
from f_toy.engine.dit_sample import sample
from f_toy.evaluation.checkpoints import load_dit, load_vae

from cond_eval.dit_eval import run_dit_eval


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--jdm", default=JDM, help="path to the Task_specific_JDM checkout")
    parser.add_argument("--dit-run", required=True)
    parser.add_argument("--dit-which", choices=["best", "last"], default="best")
    parser.add_argument("--vae-run", default=None,
                        help="default: whichever VAE this DiT records being trained against")
    parser.add_argument("--vae-which", choices=["best", "last"], default=None)
    parser.add_argument("--num-condition-frames", type=int, default=None,
                        help="leading LATENT frames used as ground-truth context "
                             "(default: this run's DIT_MIN_COND_FRAMES)")
    parser.add_argument("--steps", type=int, default=50, help="Euler integration steps")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20,
                        help="print progress every N batches")
    parser.add_argument("--no-fvd", action="store_true",
                        help="skip FVD16, which needs the pinned I3D and so a network fetch")
    parser.add_argument("--i3d-path", default=str(PROJECT / ".runtime" / "i3d" / "i3d_torchscript.pt"),
                        help="where the SHA256-pinned FVD I3D lives, downloaded there on first "
                             "use. Defaults inside this project rather than f_toy's own "
                             "repo-relative I3D_PATH, which would write into Task_specific_JDM")
    parser.add_argument("--pose-module", default=None,
                        help="'module:factory' returning a pose_fn(videos) -> (xy, heading); "
                             "omit to compute the visual metrics only")
    parser.add_argument("--no-pose-floor", action="store_true",
                        help="skip running the pose model on the real clips")
    args = parser.parse_args()
    if not args.dataset_dir:
        parser.error("--dataset-dir is required (path to the held-out test set)")

    device = args.device or get_device()
    ckpt_dir = Path(args.ckpt_dir) / (args.dataset or "cubetoy")

    # Sampling draws fresh noise, so two identical evals differ. Seed it: the
    # comparisons this feeds are paired across conditions, and an unseeded
    # sampler puts sampling variance on top of the difference being measured.
    seed = 0 if args.seed is None else args.seed
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    print(f"sampling seed: {seed}")

    dit, dit_cfg = load_dit(ckpt_dir / args.dit_run, args.dit_which, device)
    print(f"loaded DiT {ckpt_dir / args.dit_run / (args.dit_which + '.pt')} "
          f"({sum(p.numel() for p in dit.parameters()):,} params)")

    vae_run = args.vae_run or dit_cfg.get("VAE_RUN")
    if not vae_run:
        parser.error("this DiT checkpoint does not record its VAE -- pass --vae-run")
    vae_which = args.vae_which or dit_cfg.get("VAE_WHICH", "best")
    vae_run_dir = ckpt_dir / vae_run
    vae, _ = load_vae(vae_run_dir, vae_which, device)
    latent_stats = torch.load(vae_run_dir / f"{vae_which}.pt", map_location="cpu",
                              weights_only=False)["latent_stats"]
    print(f"loaded VAE {vae_run_dir / (vae_which + '.pt')} "
          f"({sum(p.numel() for p in vae.parameters()):,} params)")

    dataset = get_dataset(dit_cfg, args)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))
    test_loader = DataLoader(dataset, batch_size=args.batch or dit_cfg["BATCH"],
                             shuffle=False, num_workers=0)
    print(f"test set: {len(dataset)} samples from {args.dataset_dir}")

    num_condition_frames = args.num_condition_frames or dit_cfg.get("DIT_MIN_COND_FRAMES", 1)

    def sample_fn(videos, shape_z, condition_latent, num_cond, target_frames):
        # videos unused: a plain DiT has no r branch, and everything it needs is
        # already in condition_latent. It is in the contract for a conditioned
        # closure's sake -- see cond_eval/dit_eval.py.
        return sample(dit, vae, latent_stats, shape_z, num_steps=args.steps, device=device,
                      condition_latent=condition_latent, num_condition_frames=num_cond,
                      target_frames=target_frames)

    pose_fn = None
    if args.pose_module:
        mod_name, _, factory = args.pose_module.partition(":")
        pose_fn = getattr(importlib.import_module(mod_name), factory or "build_pose_fn")(device)
        print(f"pose model: {args.pose_module}")

    eval_dir = Path(args.eval_dir) if args.eval_dir else ckpt_dir / args.dit_run / "eval"
    run_dit_eval(sample_fn, vae, latent_stats, test_loader, dit_cfg, device, eval_dir,
                 num_condition_frames=num_condition_frames, steps=args.steps,
                 run=args.dit_run, which=args.dit_which, dataset_dir=args.dataset_dir,
                 extra={"seed": seed}, log_every=args.log_every,
                 fvd=not args.no_fvd, i3d_path=args.i3d_path, pose_fn=pose_fn,
                 pose_floor=not args.no_pose_floor)


if __name__ == "__main__":
    main()
