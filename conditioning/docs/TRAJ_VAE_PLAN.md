# Traj-VAE v2: same encoder, new objective

**Goal.** Keep the latent $r$ of V1 ($16 \times 9 \times 12 \times 12$ on the grid of $z$ for a 33-frame clip) and change what it is trained to represent, so that every latent frame describes what the ego and the scene are doing at that time and is meaningful from one context frame onward.

**Why the architecture stays.** At $k=2$ V1 separated from the null condition N (11% ADE cut against 26 to 30% for the ViTs), so the encoder-plus-head design carries driving signal. What failed at $k=1$ was the objective: one clip-level waypoint target, learned from 16 context frames and then used with one.

## Model

- **Encoder:** clone of the causal encoder of `vae_miniwan_womd`, exactly as V1 cloned `vae_50e24b`. $r$ is its `mu`, unchanged in shape and normalisation.
- **Heads:** used in training only and discarded; V outputs $r$ alone.
- **Learning rates:** encoder $5 \times 10^{-5}$, heads $10^{-3}$, as in V1.

## Targets, all from `womd_7hz_f33` labels

| loss | read from | target | weight |
|---|---|---|---|
| ego motion | $r_t$, pooled | $\Delta x, \Delta y, \Delta \psi$, speed within the 4 raw frames of latent frame $t$, and over the next 0.57 s | 1 |
| waypoints | $r_0 \dots r_k$, $k \sim \{1, 2, 3\}$ | ego path over the next 4 s; 6 hypotheses, loss on the closest | 1 |
| agents | $r_t$, spatial map | occupancy of visible agents in the image plane, on the $12 \times 12$ grid | 0.5 |
| scene context | $r_t$, pooled | ego-lane curvature, intersection flag, traffic-light state | 0.25 |
| KL | $r$ | $\mathcal{N}(0, I)$ | $10^{-4}$ |

Causality makes varying context free: $r_t$ never sees later frames, so a random $k$ trains V for 1, 2 or 3 context latent frames with no masking.

No distillation from D, T or G, so the comparison between them stays clean.

## Training

`womd_7hz_f33` train split, best checkpoint by summed validation loss on val. Per-channel $r$ statistics (frame 0 and the rest) are computed from train clips afterwards and stored in the checkpoint, as V1 does.

## Checks on val, before any DiT

| check | passes if |
|---|---|
| ego motion from $r$ per frame | close to the pose probe on real frames |
| waypoint ADE at $k = 1, 2, 3$ | falls with $k$ and beats V1 at every $k$ |
| agent occupancy | clearly above a frame-0-only baseline |
| $R^2$ of a linear map $z \to r$ | low |
| active channels | most of 16 |
| $r$-only DiT samples, decoded by the heads | plausible trajectories |

## Evaluation

1. Stage-1 conditioning screen on `womd_7hz_f33`: V2 against N, D, T, G.
2. Joint diffusion of $r$ with $z$, the setting V is designed for.

V learns from 40,000 clips of in-domain labels while D, T and G carry large external pretraining; a loss to T means driving supervision at this scale does not match that pretraining, not that driving objectives do not help.
