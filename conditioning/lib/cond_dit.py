"""Wire conditioning tokens into MiniDiT's cross-attention.

Simpler than it first looked. ``f_toy.engine.dit.build_dit`` already reads
``DIT_CROSSATTN_DIM`` into ``crossattn_emb_channels``, and ``Attention`` takes a
separate ``context_dim`` for ``k_proj``/``v_proj``, so a 1024-d Drive-JEPA token
stream needs no projection layer: set that config key to the encoder's width and
the DiT's own k/v projections do the mapping, trained from scratch along with
everything else.

Two things still have to be added around it.

**A LayerNorm**, because raw ViT tokens are badly scaled for a freshly
initialised ``k_proj``: Phase 0 measured token norms near 90, and the softmax
would saturate into a delta on the first step.

**A learned positional embedding**, because ``Attention`` applies RoPE only when
``is_selfattn``, so cross-attention context arrives as an unordered bag. That is
right for text and wrong for a dense spatial-temporal grid, where which token
came from which place and time is the information worth having.

The embedding is added after the norm rather than before, so normalisation does
not partially erase it.

Condition **N** needs none of this: passing ``r_tokens=None`` falls through to
``MiniDiT``'s own learned ``null_context``, so the null baseline and the
conditioned runs share one code path and one training script.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CrossAttnConditioner", "ConditionedDiT", "latent_cutoffs"]


def latent_cutoffs(cond_mask, temporal_compression_factor):
    """Per-sample raw-frame cutoff from a Video2World condition mask.

    ``cond_mask`` is ``(B,1,Tz,Hz,Wz)`` from
    ``f_toy.models.dit_loss.sample_condition_mask``, with each sample's first
    ``k`` latent frames set to 1. Latent frame 0 covers raw frame 0 alone and
    every later one pools ``temporal_compression_factor`` raw frames, so the
    first raw frame the model must predict is ``1 + (k-1)*tc``. This is the
    vectorised form of ``conditioning.lib.dit_eval.condition_raw_frame_cutoff``."""
    k = cond_mask[:, 0, :, 0, 0].sum(dim=1).long()
    return torch.where(k > 0, 1 + (k - 1) * temporal_compression_factor,
                       torch.zeros_like(k))


class CrossAttnConditioner(nn.Module):
    """``LayerNorm`` then a learned per-position embedding, over ``(B, N, d)``."""

    def __init__(self, dim, num_tokens):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, tokens):
        if tokens.shape[1] != self.pos.shape[1]:
            raise ValueError(f"expected {self.pos.shape[1]} tokens, got {tokens.shape[1]}")
        return self.norm(tokens) + self.pos


class ConditionedDiT(nn.Module):
    """``MiniDiT`` plus the conditioner, so the training loop sees one module.

    ``conditioner=None`` is condition **N**: ``crossattn_emb`` stays ``None`` and
    ``MiniDiT`` substitutes its own ``null_context``."""

    def __init__(self, dit, conditioner=None):
        super().__init__()
        self.dit = dit
        self.conditioner = conditioner

    def forward(self, xt, t, r_tokens=None, condition_mask_B_1_T_H_W=None):
        ctx = None
        if self.conditioner is not None:
            if r_tokens is None:
                raise ValueError("this DiT was built with a conditioner but got no r tokens")
            ctx = self.conditioner(r_tokens.to(xt.dtype))
        return self.dit(xt, t, crossattn_emb=ctx,
                        condition_mask_B_1_T_H_W=condition_mask_B_1_T_H_W)
