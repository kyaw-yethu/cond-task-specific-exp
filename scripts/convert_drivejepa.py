#!/usr/bin/env python
"""Convert a released Drive-JEPA or V-JEPA2 checkpoint into an f_toy
``VJepa2Encoder`` state dict, under this project's checkpoint tree.

`Task_specific_JDM/scripts/import_drive_jepa.py` cannot do it for the task
checkpoint: its ``ft`` variant strips ``agent._model.image_encoder.``, which
matches a different Drive-JEPA release and yields zero of the expected 292
tensors here. `phantom-drivejepa`'s Phase 0 recorded the prefixes that do work,
verified on these exact files at 292 of 292 tensors with zero missing or
unexpected keys:

| label | file | state key | prefix |
|---|---|---|---|
| **D** domain | `vitl_merge_3dataset_e50.pt` | `target_encoder` | `module.backbone.` |
| **T** task | `drive_jepa_perception_based_agent_vitl_v2.ckpt` | `state_dict` | `agent._pad_model._backbone.img_backbone.` |
| **G** generic | `vjepa2-vith-fpc64-256` | `target_encoder` | (none) |

Rather than trusting a label, this tries every known combination, reports how
many tensors each matches, and uses the one that fits. A clean load is the
acceptance criterion: a silently partial load would look like a weak
representation rather than a broken one, which is the most expensive kind of
mistake available here.

Usage:
    python scripts/convert_drivejepa.py --src <ckpt> --label T \\
        --run drivejepa_T_vitl256 --img 256 --ckpt-dir <proj>/checkpoints
"""
import argparse
import sys
from pathlib import Path


def _bootstrap(jdm):
    if not Path(jdm, "f_toy").is_dir():
        sys.exit(f"no f_toy under {jdm}; pass --jdm")
    sys.path.insert(0, str(Path(jdm).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CANDIDATES = {
    "D": [("target_encoder", "module.backbone."), ("target_encoder", "module."),
          ("target_encoder", "")],
    "T": [("state_dict", "agent._pad_model._backbone.img_backbone."),
          ("state_dict", "agent._model.image_encoder."), ("state_dict", "")],
    "G": [("target_encoder", ""), ("target_encoder", "module.backbone."),
          ("module", "")],
}
EXPECTED = {1024: 292, 1280: 388}   # patch_embed 2 + depth*12 + final norm 2


def strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jdm", default="third_party/Task_specific_JDM")
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="waymo")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--variant", choices=["large", "huge"], default="large",
                    help="ViT width: D and T are large, G is huge")
    ap.add_argument("--frames", type=int, default=6,
                    help="the context width this will be fed; nominal only, the ViT "
                         "runs on any grid")
    args = ap.parse_args()

    _bootstrap(args.jdm)
    import torch
    from cond_eval.r_sources import build_vjepa_encoder

    print(f"loading {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"top-level keys: {sorted(k for k in ckpt)[:8]}")

    enc = build_vjepa_encoder(num_frames=args.frames, img_size=args.img,
                              patch_size=16, tubelet_size=2, variant=args.variant)
    target = enc.vit.state_dict()
    want = EXPECTED.get(enc.vit.dim, len(target))
    print(f"target: VJepa2Encoder vit dim {enc.vit.dim}, {len(target)} tensors "
          f"(expect {want})")

    print("\ncandidate prefixes:")
    best = None
    for key, prefix in CANDIDATES[args.label]:
        if key not in ckpt:
            print(f"  {key!r:16s} {prefix!r:48s} state key absent")
            continue
        sd = strip(ckpt[key], prefix)
        sd = {k: v for k, v in sd.items() if not k.startswith("predictor")}
        shared = sum(1 for k, v in sd.items() if k in target and v.shape == target[k].shape)
        print(f"  {key!r:16s} {prefix!r:48s} {len(sd):4d} keys, {shared:4d} match by name+shape")
        if best is None or shared > best[2]:
            best = (key, prefix, shared, sd)

    if best is None or best[2] == 0:
        sys.exit("no candidate matched a single tensor; the layout is unknown")
    key, prefix, shared, sd = best
    print(f"\nusing {key!r} + {prefix!r}: {shared} tensors")

    missing, unexpected = enc.vit.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "pos_embed" not in m]   # RoPE: not a parameter
    print(f"load: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print(f"  first missing:    {missing[:4]}")
        print(f"  first unexpected: {unexpected[:4]}")
        sys.exit("refusing to save a partial load")
    if shared != want:
        sys.exit(f"loaded {shared} tensors, expected {want}")

    n_par = sum(p.numel() for p in enc.vit.parameters())
    out_dir = Path(args.ckpt_dir) / args.dataset / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"VJEPA_SOURCE": Path(args.src).name, "VJEPA_LABEL": args.label,
           "VJEPA_PATCH": 16, "VJEPA_TUBELET": 2, "VJEPA_DIM": enc.vit.dim,
           "VJEPA_IMG": args.img, "VJEPA_FRAMES": args.frames,
           "VJEPA_VARIANT": args.variant,
           "VJEPA_STATE_KEY": key, "VJEPA_PREFIX": prefix}
    torch.save({"cfg": cfg, "tgt_enc": enc.state_dict()}, out_dir / "last.pt")
    print(f"\nsaved {out_dir / 'last.pt'}  ({n_par:,} ViT params)")


if __name__ == "__main__":
    main()
