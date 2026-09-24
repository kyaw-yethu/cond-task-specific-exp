"""Conditioning sources: turn a clip's context frames into cross-attention tokens.

Each source answers one question: given a batch of clips and each clip's
raw-frame cutoff, what tokens does the DiT get to attend to? The contract is

    encode(videos, cutoff_B) -> (B, N, d) tokens,  plus .grid == (t, h, w)
                                                   and  .dim == d

with ``N == t*h*w`` and the grid fixed across the batch so a learned positional
embedding can be added (``conditioning.lib.cond_dit``).

**No future leakage, by construction.** A source only ever sees raw frames
strictly before a clip's own cutoff. Frames at or after it are replaced by the
last context frame, which is the rule ``f_toy.engine.phantom.RSource`` already
uses. So a sample with k=1 sees frame 0 repeated, and one with k=2 sees frames
0..4 with frame 4 repeated.

**Why the window is fixed at the widest context, not each sample's own.**
``DIT_MIN/MAX_COND_FRAMES`` are sampled per sample, so k varies within a batch,
and a per-sample window would give per-sample token counts that cannot be
batched into one encoder forward or attended to without a key-padding mask that
``f_toy``'s ``Attention`` does not have. Every sample therefore uses the k=2
window of 6 raw frames, with each sample's own frames repeated past its own
cutoff. Uniform 768 tokens at 256px, one forward, no leakage, no mask. It costs
1.5x what a per-sample window would, since a k=1 sample carries 768 tokens of
repeated content rather than 256.

**Resolution.** The ViT candidates run at 256x256, their own training
resolution, via bicubic upsample from 96. ``VJepa2Encoder`` applies ImageNet
normalisation internally, so nothing is done here beyond the resize.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["CondSource", "VitCondSource", "TrajVaeCondSource", "build_cond_source",
           "COND_SOURCES", "context_window", "build_vjepa_encoder", "VIT_VARIANTS"]

COND_SOURCES = ("drivejepa", "vjepa2", "traj_vae")

# --------------------------------------------------------------------------
# f_toy's VJepa2Encoder hardcodes ViT-L (dim 1024, depth 24). Condition G is
# V-JEPA2 ViT-H, so its VisionTransformer has to be rebuilt at the right width.
# --------------------------------------------------------------------------
VIT_VARIANTS = {
    "large": dict(dim=1024, depth=24, num_heads=16, n_tensors=292),
    "huge": dict(dim=1280, depth=32, num_heads=16, n_tensors=388),
}


def build_vjepa_encoder(num_frames, img_size, patch_size=16, tubelet_size=2,
                        variant="large", data_channel_order="rgb"):
    """A ``VJepa2Encoder`` whose ViT is the requested variant."""
    from f_toy.models.vjepa2 import VisionTransformer, VJepa2Encoder

    spec = VIT_VARIANTS[variant]
    enc = VJepa2Encoder(num_frames=num_frames, img_size=img_size,
                        patch_size=patch_size, tubelet_size=tubelet_size,
                        data_channel_order=data_channel_order)
    if enc.vit.dim != spec["dim"]:
        enc.vit = VisionTransformer(dim=spec["dim"], depth=spec["depth"],
                                    num_heads=spec["num_heads"],
                                    patch_size=patch_size, tubelet_size=tubelet_size)
        enc.dim = spec["dim"]
    return enc



def context_window(videos, cutoff_B, width):
    """The leading ``width`` raw frames, with each clip's frames at or after its
    own ``cutoff`` replaced by that clip's last context frame.

    ``videos`` (B,3,T,H,W), ``cutoff_B`` (B,) of raw-frame cutoffs as
    ``condition_raw_frame_cutoff`` computes them. Returns (B,3,width,H,W)."""
    B, C, T, H, W = videos.shape
    if width > T:
        raise ValueError(f"context width {width} exceeds clip length {T}")
    pos = torch.arange(width, device=videos.device)[None]            # (1,width)
    idx = torch.minimum(pos, (cutoff_B - 1)[:, None].to(pos.device))  # clamp to own context
    idx = idx.clamp_(min=0)
    return videos.gather(2, idx[:, None, :, None, None].expand(B, C, width, H, W))


class CondSource:
    """Base. Subclasses set ``kind``, ``dim``, ``grid`` and implement
    ``_encode(window) -> (B, N, d)``."""

    kind = None
    dim = None
    grid = None          # (t, h, w); N == t*h*w
    context_width = None

    @property
    def num_tokens(self):
        t, h, w = self.grid
        return t * h * w

    @torch.no_grad()
    def encode(self, videos, cutoff_B):
        window = context_window(videos, cutoff_B, self.context_width)
        tokens = self._encode(window)
        if tokens.shape[1] != self.num_tokens:
            raise RuntimeError(f"{self.kind}: got {tokens.shape[1]} tokens, "
                               f"grid {self.grid} implies {self.num_tokens}")
        return tokens

    def _encode(self, window):
        raise NotImplementedError


class VitCondSource(CondSource):
    """Drive-JEPA (domain or task) and V-JEPA2, all ``VJepa2Encoder``.

    Tokens come back time-major then row then column, matching the (t, h, w)
    grid, which is what the positional embedding assumes."""

    def __init__(self, encoder, kind, context_width=6, img_size=256, device="cuda"):
        self.kind = kind
        self.context_width = context_width
        self.img_size = img_size
        self.model = encoder.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        vit = self.model.vit
        self.dim = vit.dim
        s = img_size // vit.patch_size
        if context_width % vit.tubelet_size:
            raise ValueError(f"context_width {context_width} is not a multiple of "
                             f"tubelet {vit.tubelet_size}")
        self.grid = (context_width // vit.tubelet_size, s, s)

    def _encode(self, window):
        B, C, T, H, W = window.shape
        if (H, W) != (self.img_size, self.img_size):
            window = F.interpolate(
                window.transpose(1, 2).reshape(B * T, C, H, W),
                size=(self.img_size, self.img_size), mode="bicubic", align_corners=False,
            ).clamp(0, 1).reshape(B, T, C, self.img_size, self.img_size).transpose(1, 2)
        return self.model(window).float()


class TrajVaeCondSource(CondSource):
    """Condition V: the Traj-VAE latent, flattened to tokens.

    Unlike the ViTs this needs no resize, since its backbone is a clone of the
    VAE encoder and already lives at 96px on the VAE's own latent grid. It is
    causal, so encoding the 6-frame window leaks nothing by construction, and
    the repeat rule above is applied anyway for consistency."""

    kind = "traj_vae"

    def __init__(self, model, mean, std, context_width=6, img_size=96, device="cuda"):
        self.context_width = context_width
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        t = model.latent_num_frames(context_width)
        s = img_size // model.spatial_compression_factor
        self.grid = (t, s, s)
        self.mean, self.std = mean, std
        self.dim = mean.shape[1] if mean.ndim > 1 else int(mean.numel())

    def _encode(self, window):
        from f_toy.engine.dit import encode_traj_vae
        z = encode_traj_vae(self.model, window, self.mean, self.std).float()
        B, C = z.shape[0], z.shape[1]
        return z.reshape(B, C, -1).transpose(1, 2)      # (B, N, C), time-major


def build_cond_source(kind, ckpt_dir, device, context_width=6, img_size=256,
                      run=None, which="last"):
    """Load the source ``kind`` names. ``ckpt_dir`` is this project's
    ``checkpoints/<dataset>`` directory."""
    from pathlib import Path

    ckpt_dir = Path(ckpt_dir)
    if kind in ("drivejepa", "vjepa2"):
        ck = torch.load(ckpt_dir / run / f"{which}.pt", map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {})
        enc = build_vjepa_encoder(num_frames=context_width, img_size=img_size,
                                  patch_size=cfg.get("VJEPA_PATCH", 16),
                                  tubelet_size=cfg.get("VJEPA_TUBELET", 2),
                                  variant=cfg.get("VJEPA_VARIANT", "large"))
        missing, unexpected = enc.load_state_dict(ck["tgt_enc"], strict=False)
        # A clean load is the acceptance criterion: a silently partial load would
        # look like a weak representation rather than a broken one.
        if missing or unexpected:
            raise RuntimeError(f"{run}: {len(missing)} missing and {len(unexpected)} "
                               f"unexpected keys; first missing {missing[:3]}")
        return VitCondSource(enc, kind, context_width, img_size, device)

    if kind == "traj_vae":
        from f_toy.engine.dit import stack_traj_stats
        from f_toy.evaluation.checkpoints import load_vae_traj
        # load_vae_traj returns (model, cfg, traj_stats): the normalisation stats
        # live in this checkpoint rather than a separate traj_pca.pt.
        model, _tcfg, traj_stats = load_vae_traj(ckpt_dir / run, which, device)
        t = model.latent_num_frames(context_width)
        mean, std = stack_traj_stats(traj_stats, t, device)
        return TrajVaeCondSource(model, mean, std, context_width, img_size=96, device=device)

    raise ValueError(f"unknown conditioning source {kind!r} (known: {', '.join(COND_SOURCES)})")
