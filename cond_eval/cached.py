"""Precomputed VAE latents and conditioning tokens, as memmaps.

Fixing k=2 makes both quantities pure functions of the clip, so both can be
computed once instead of every epoch. The profile that motivated this, at batch
8 on one 3090:

    VAE encode_latents                105 ms
    r encoder (ViT-L, 768 tokens)     292 ms   <- 60% of the step
    DiT forward + backward + step      92 ms

The encoder cannot be made much faster in place: it already uses
`F.scaled_dot_product_attention`, and it is launch-overhead bound rather than
FLOP bound, since the RoPE rotation runs a Python loop over three axes for q and
k in each of 24 blocks. So the fix is to stop recomputing it. With both cached
the step is the DiT alone, about 5.9x faster, and the 17.7 GB of video is never
read during training either.

**The cost of fixing k.** `f_toy`'s `sample_condition_mask` draws k per sample
from `DIT_MIN/MAX_COND_FRAMES`, and a cache cannot depend on a per-sample draw:
holding both k=1 and k=2 token sets would need 84 GB against 75 GB free. Fixing
k=2 therefore changes the training distribution away from the recipe that
produced `dit10m32b100e`, which means **condition N must be retrained with this
same fixed k** rather than reusing that checkpoint. k=2 is also the more
informative window, since k=1 shows the encoder a single distinct frame repeated.

Sizes at 40,000 train clips, fp16: latents 921 MB, ViT-L tokens 62.9 GB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CachedCondDataset", "cache_meta_path", "write_cache_meta", "read_cache_meta"]


def cache_meta_path(cache_dir):
    return Path(cache_dir) / "cache_meta.json"


def write_cache_meta(cache_dir, **kw):
    cache_meta_path(cache_dir).write_text(json.dumps(kw, indent=2))


def read_cache_meta(cache_dir):
    p = cache_meta_path(cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"no cache_meta.json in {cache_dir}; build it with "
                                f"scripts/build_cache.py")
    return json.loads(p.read_text())


class CachedCondDataset(Dataset):
    """Yields ``(x0, r)`` straight from disk: the normalised VAE latent and the
    conditioning tokens for one clip. ``r`` is ``None`` for condition N.

    Both files are opened as memmaps per worker, never loaded whole: the token
    cache is 63 GB against 480 GB of RAM, so it would fit, but paging it in per
    worker would serialise the first epoch behind 63 GB of reads.
    """

    def __init__(self, cache_dir, split, with_r=True):
        self.dir = Path(cache_dir)
        self.meta = read_cache_meta(cache_dir)
        if split not in self.meta["splits"]:
            raise KeyError(f"split {split!r} not in this cache: {list(self.meta['splits'])}")
        self.split = self.meta["splits"][split]
        self.with_r = with_r and self.split.get("r_file") is not None
        self._x0 = self._r = None

    def __len__(self):
        return int(self.split["n"])

    def _open(self):
        # Lazily, so each DataLoader worker gets its own handle rather than
        # inheriting one across a fork.
        if self._x0 is None:
            self._x0 = np.load(self.dir / self.split["x0_file"], mmap_mode="r")
        if self.with_r and self._r is None:
            self._r = np.load(self.dir / self.split["r_file"], mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        x0 = torch.from_numpy(np.asarray(self._x0[i], dtype=np.float32))
        if not self.with_r:
            return x0, 0
        return x0, torch.from_numpy(np.asarray(self._r[i], dtype=np.float32))
