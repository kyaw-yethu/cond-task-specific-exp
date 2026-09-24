"""Traj-VAE v2: V1 architecture, new objective (see TRAJ_VAE_PLAN.md).

The encoder is a clone of a trained MiniWan, exactly as V1 cloned vae_50e24b, and
r is its ``mu`` on the grid of z. What changes is what the heads ask of r:

    motion    per latent frame: ego motion over its own 4 raw frames and over the
              next 0.57 s, so every latent frame means something on its own
    waypoint  from r_0..r_k with k drawn from 1..3: the ego path over the next 4 s,
              6 hypotheses with the loss on the closest, so r is not pushed toward
              an averaged future
    occupancy per latent frame: which of the 12x12 image cells hold a visible agent
    scene     per latent frame: lane curvature, intersection flag, traffic light

Causality does the rest: r_t never sees frames after its own chunk, so drawing k at
random trains 1, 2 or 3 context latent frames without any masking.

The heads exist only during training; V outputs r alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrajV2", "traj_losses", "motion_targets", "waypoint_targets", "chunk_end"]

N_TL = 5          # tl_state -1 none, 0 unknown, 1 stop, 2 caution, 3 go
FUT = 28          # 4 s at 7 Hz


def chunk_end(T, tc=4):
    """Raw frame index each latent frame ends on: 0, 4, 8, ... for a causal 4x VAE."""
    Tz = 1 + (T - 1) // tc
    return torch.arange(Tz) * tc


def _rot(dx, dy, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def motion_targets(pos, yaw, speed, ends, back=4, fwd=4):
    """Per latent frame: motion over the 4 raw frames ending at its chunk end, and
    over the next 4, both in the ego frame at that end. (B,Tz,8)."""
    B = pos.shape[0]
    T = pos.shape[1]
    e = ends.to(pos.device)
    b0 = (e - back).clamp(min=0)
    f1 = (e + fwd).clamp(max=T - 1)
    g = lambda t, idx: t[:, idx]
    yaw_e = g(yaw, e)
    out = []
    for lo, hi in ((b0, e), (e, f1)):
        dx, dy = _rot(g(pos, hi)[..., 0] - g(pos, lo)[..., 0],
                      g(pos, hi)[..., 1] - g(pos, lo)[..., 1], yaw_e)
        dyaw = torch.atan2(torch.sin(g(yaw, hi) - g(yaw, lo)), torch.cos(g(yaw, hi) - g(yaw, lo)))
        out += [dx, dy, dyaw]
    out.append(g(speed, e))
    out.append(0.5 * (g(speed, e) + g(speed, f1)))
    return torch.stack(out, dim=-1).reshape(B, len(e), 8)


def waypoint_targets(pos_all, yaw_all, valid_all, cutoff):
    """Ego path over the next FUT steps after raw frame ``cutoff``, in the ego frame
    at ``cutoff``. ``pos_all`` (B, 33+28, 2) is the clip followed by its future.
    Returns (B, FUT, 2) and a (B, FUT) validity mask."""
    B = pos_all.shape[0]
    idx = cutoff + 1 + torch.arange(FUT, device=pos_all.device)
    idx = idx.clamp(max=pos_all.shape[1] - 1)
    p = pos_all[:, idx]
    v = valid_all[:, idx] * (cutoff + 1 + torch.arange(FUT, device=pos_all.device) <= pos_all.shape[1] - 1).float()
    p0 = pos_all[:, cutoff:cutoff + 1]
    y0 = yaw_all[:, cutoff:cutoff + 1]
    dx, dy = _rot(p[..., 0] - p0[..., 0], p[..., 1] - p0[..., 1], y0)
    return torch.stack([dx, dy], dim=-1), v


class TrajV2(nn.Module):
    def __init__(self, vae, head_width=64, dropout=0.1, n_hyp=6):
        super().__init__()
        self.vae = vae
        self.vae.decoder.requires_grad_(False)     # never called, as in VaeTrajCNN
        self.vae.conv2.requires_grad_(False)
        self.n_hyp = n_hyp
        z = vae.z_dim
        W = head_width
        self.trunk = nn.Sequential(
            nn.Conv3d(z, W, 3, padding=1), nn.SiLU(),
            nn.Dropout3d(dropout), nn.Conv3d(W, W, 3, padding=1), nn.SiLU())
        self.motion = nn.Sequential(nn.Linear(W, 128), nn.SiLU(), nn.Linear(128, 8))
        self.occ = nn.Conv3d(W, 1, 1)
        self.scene = nn.Sequential(nn.Linear(W, 64), nn.SiLU(), nn.Linear(64, 2 + N_TL))
        self.k_emb = nn.Embedding(4, W)
        self.way = nn.Sequential(nn.Linear(W, 256), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(256, n_hyp * (FUT * 2 + 1)))

    @property
    def temporal_compression_factor(self):
        return self.vae.temporal_compression_factor

    @property
    def spatial_compression_factor(self):
        return self.vae.spatial_compression_factor

    def latent_num_frames(self, T):
        return self.vae.latent_num_frames(T)

    def backbone_parameters(self):
        return list(self.vae.encoder.parameters()) + list(self.vae.conv1.parameters())

    def head_parameters(self):
        return (list(self.trunk.parameters()) + list(self.motion.parameters())
                + list(self.occ.parameters()) + list(self.scene.parameters())
                + list(self.way.parameters()) + list(self.k_emb.parameters()))

    @torch.no_grad()
    def latent_repr(self, video):
        """r as every downstream user sees it: the clean mu on the grid of z."""
        mu, _ = self.vae.encode(video)
        return mu

    def forward(self, video, k):
        """``k``: latent context frames, 1..3 (one value for the batch)."""
        mu, log_var = self.vae.encode(video)
        log_var = log_var.clamp(-30.0, 20.0)
        r = self.vae.reparameterize(mu, log_var) if self.training else mu
        h = self.trunk(r)                                   # (B, W, Tz, 12, 12)
        pooled = h.mean(dim=(3, 4)).transpose(1, 2)          # (B, Tz, W)
        ctx = pooled[:, :k].mean(1) + self.k_emb(torch.full((h.shape[0],), k, device=h.device))
        w = self.way(ctx).view(h.shape[0], self.n_hyp, FUT * 2 + 1)
        return dict(motion=self.motion(pooled), occ=self.occ(h)[:, 0],
                    scene=self.scene(pooled), way=w[..., :FUT * 2].view(-1, self.n_hyp, FUT, 2),
                    way_logit=w[..., -1], mu=mu, log_var=log_var)


def traj_losses(out, tgt, weights, kl_w=1e-4):
    """Scalar loss plus a dict of parts. ``tgt`` holds motion, way, way_valid,
    occ, curvature, intersection, tl."""
    p = {}
    p["motion"] = F.smooth_l1_loss(out["motion"], tgt["motion"], beta=0.2)

    d = ((out["way"] - tgt["way"][:, None]) ** 2).sum(-1).sqrt()      # (B, K, FUT)
    v = tgt["way_valid"][:, None]
    per_hyp = (d * v).sum(-1) / v.sum(-1).clamp_min(1.0)
    best = per_hyp.argmin(dim=1)
    p["way"] = per_hyp.gather(1, best[:, None]).mean()
    p["way_cls"] = F.cross_entropy(out["way_logit"], best.detach())

    p["occ"] = F.binary_cross_entropy_with_logits(out["occ"], tgt["occ"])
    p["curv"] = F.smooth_l1_loss(out["scene"][..., 0], tgt["curvature"], beta=0.02)
    p["inter"] = F.binary_cross_entropy_with_logits(out["scene"][..., 1], tgt["intersection"])
    p["tl"] = F.cross_entropy(out["scene"][..., 2:].reshape(-1, N_TL), tgt["tl"].reshape(-1))

    mu, lv = out["mu"], out["log_var"]
    p["kl"] = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean()

    total = (weights["motion"] * p["motion"]
             + weights["way"] * (p["way"] + 0.1 * p["way_cls"])
             + weights["occ"] * p["occ"]
             + weights["scene"] * (p["curv"] + p["inter"] + p["tl"])
             + kl_w * p["kl"])
    return total, {k: float(v.detach()) for k, v in p.items()}
