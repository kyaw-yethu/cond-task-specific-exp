# Stage 1 results: driving representation as conditioning

One seed. Five conditions, each a MiniDiT trained from scratch for 100 epochs at
effective batch 32 with $k$ fixed at 2, differing only in what feeds the
cross-attention. Evaluated on all 4,000 held-out test clips, seed 0, 50 Euler
steps. Design and rationale in `STAGE1_PLAN.md`; the trajectory extractor in
`PROBE.md`.

Clips evaluated: **4000**. Context frames raw 0 to 4 are ground truth in every
condition; raw 5 to 15 are generated.

Beside each metric name, ↑ means higher is better and ↓ means lower is
better, and **bold** marks the best value in that column, read in the
arrow's direction.

## 1. Visual axis

| condition | tokens | FVD16 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | epoch | val_flow ↓ |
|---|---|---|---|---|---|---|---|
| **N** null context | - | 211.02 | 26.671 | 0.7607 | 0.0681 | 85 | - |
| **D** Drive-JEPA domain | 768 x 1024 | **208.56** | 27.159 | 0.7754 | 0.0572 | 85 | - |
| **T** Drive-JEPA task | 768 x 1024 | 209.63 | 27.176 | 0.7763 | 0.0566 | 85 | - |
| **V** Traj-VAE | 432 x 16 | 214.37 | 26.749 | 0.7629 | 0.0662 | 85 | - |
| **G** V-JEPA2 ViT-H (generic) | 768 x 1280 | 209.06 | **27.287** | **0.7781** | **0.0548** | 85 | - |

FVD16 is the primary visual metric. PSNR, SSIM and LPIPS are over the generated
frame range only; FVD16 covers the whole clip, which is the convention the
video-prediction literature uses and which `cond_eval/dit_eval.py` documents.

## 2. Trajectory axis

Read against the probe's VAE round-trip floor of **0.2739 m**
(`PROBE.md` section 5). Differences below that floor are not readable.

| condition | ADE (m) ↓ | FDE (m) ↓ | DTW ↓ | heading (deg) ↓ | composite ↑ |
|---|---|---|---|---|---|
| **N** null context | 0.3395 | 1.0090 | 3.481 | 0.252 | 0.6769 |
| **D** Drive-JEPA domain | 0.2510 | 0.7401 | 2.716 | **0.170** | 0.6811 |
| **T** Drive-JEPA task | 0.2465 | 0.7264 | 2.696 | 0.172 | **0.6818** |
| **V** Traj-VAE | 0.3016 | 0.8999 | 3.163 | 0.246 | 0.6748 |
| **G** V-JEPA2 ViT-H (generic) | **0.2379** | **0.6973** | **2.622** | 0.176 | 0.6817 |

ADE is the primary trajectory metric. DTW absorbs a shift along the direction
of travel, so a condition with the right path at the wrong speed shows in ADE
and not in DTW. The composite is a guard against degenerate motion rather than
a discriminator, being near its ceiling on real motion.

## 3. Paired bootstrap against N

10,000 resamples over the shared test clips. A negative delta favours the
condition for an error metric. `p` is the fraction of resamples whose sign
differs from the observed delta.

**PSNR ↑**, higher is better

| condition | delta vs N | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|
| **D** Drive-JEPA domain | +0.4883 | [+0.4617, +0.5153] | 0.0000 | yes |
| **T** Drive-JEPA task | +0.5055 | [+0.4796, +0.5317] | 0.0000 | yes |
| **V** Traj-VAE | +0.0784 | [+0.0547, +0.1021] | 0.0000 | yes |
| **G** V-JEPA2 ViT-H (generic) | +0.6158 | [+0.5891, +0.6422] | 0.0000 | yes |

**LPIPS ↓**, lower is better

| condition | delta vs N | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|
| **D** Drive-JEPA domain | -0.0109 | [-0.0114, -0.0104] | 0.0000 | yes |
| **T** Drive-JEPA task | -0.0115 | [-0.0121, -0.0110] | 0.0000 | yes |
| **V** Traj-VAE | -0.0019 | [-0.0024, -0.0015] | 0.0000 | yes |
| **G** V-JEPA2 ViT-H (generic) | -0.0133 | [-0.0138, -0.0128] | 0.0000 | yes |

**ADE ↓**, lower is better

| condition | delta vs N | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|
| **D** Drive-JEPA domain | -0.0886 | [-0.0975, -0.0798] | 0.0000 | yes |
| **T** Drive-JEPA task | -0.0931 | [-0.1019, -0.0843] | 0.0000 | yes |
| **V** Traj-VAE | -0.0380 | [-0.0465, -0.0295] | 0.0000 | yes |
| **G** V-JEPA2 ViT-H (generic) | -0.1016 | [-0.1107, -0.0929] | 0.0000 | yes |

**DTW ↓**, lower is better

| condition | delta vs N | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|
| **D** Drive-JEPA domain | -0.7646 | [-0.8478, -0.6865] | 0.0000 | yes |
| **T** Drive-JEPA task | -0.7847 | [-0.8680, -0.7047] | 0.0000 | yes |
| **V** Traj-VAE | -0.3174 | [-0.3933, -0.2440] | 0.0000 | yes |
| **G** V-JEPA2 ViT-H (generic) | -0.8593 | [-0.9445, -0.7776] | 0.0000 | yes |

FVD16 has no per-clip value, being a distance between distributions, so it
cannot be bootstrapped this way and is compared on its point estimate alone.

## 4. Qualitative comparison

Ground truth and every condition on the same clips, same seed, same context
frames. Row order is in `out/comparison/ROW_ORDER.txt`.

![sheet_clip1137_brisk](out/comparison/sheet_clip1137_brisk.png)

![sheet_clip1565_turning](out/comparison/sheet_clip1565_turning.png)

![sheet_clip3217_turning](out/comparison/sheet_clip3217_turning.png)

All 6 sheets and 6 animated versions are in
`out/comparison/`. The GIFs are the more useful of the two for judging
temporal coherence.

## 5. How to read this

**One seed, so the plan's separation rule is only partly applicable.**
`STAGE1_PLAN.md` section 8 requires a gap exceeding twice the pooled across-seed
standard deviation, which cannot be evaluated without repeats. What is available
is the paired bootstrap over clips and the probe floor, and both are reported
above. A bootstrap CI excluding zero says the ordering is stable across clips,
not that it is stable across training runs.

**Conditioning bandwidth is not controlled.** The conditions differ enormously in
how much they feed the cross-attention: V is 432 tokens of width 16, so 6,912
numbers, while D and T are 768 x 1024 and G is 768 x 1280, so 786,432 and 983,040.
That is a factor of 114 to 142. If the high-bandwidth conditions lead, capacity is
as good an explanation as driving-specificity, and the D-against-T comparison is
the one that isolates domain from task at matched bandwidth.

**$k$ is fixed at 2**, which departs from the sampled $k$ that produced
`dit10m32b100e`. That is why N was retrained here rather than reused; all five
share the fixed-$k$ distribution and so are comparable to each other, but none is
comparable to that earlier checkpoint.

