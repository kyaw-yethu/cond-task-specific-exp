# Stage 1: driving representation as conditioning, in place of text

**Question.** Replace a video DiT's text conditioning with a driving-specific
representation. Does the representation buy visual quality, trajectory quality, or
neither? Answer this before building Phantom's dual cross-attention joint denoiser,
because a representation that cannot guide through the cheap path is unlikely to repay
a coupled branch.

Video branch only. No $r$ denoising, no coupling.

---

## 1. Where this sits relative to Phantom

Phantom (arXiv:2604.08503) fine-tunes **Wan2.2-TI2V-5B** at $480\times832$ over up to
121 frames, freezing every pretrained visual-branch parameter and training only the
physics branch plus the dual cross-attention. Its physics encoder is **V-JEPA2 ViT-H
`fpc64-256`**, a generic video encoder, and it injects the text prompt and the
flow-matching timestep into **both** branches.

Two differences define this experiment. Phantom's physics encoder is generic, so
V-JEPA2 becomes the reference to beat rather than an optional control. And Phantom
*augments* text, while this *replaces* conditioning outright, which is the stronger
claim.

## 2. Two stages

| stage | stack | resolution | status |
|---|---|---|---|
| **1** | `MiniWan` VAE + `MiniDiT`, from `Task_specific_JDM` | $96\times96$, 16 frames | this plan |
| **2** | Wan2.2-TI2V-5B on nuScenes | $480\times832$ | **`STAGE2_PLAN.md`** |

Stage 1 exists because all four candidate representations are comparable there today
and a run costs hours rather than days. Stage 2 inherits whichever conditions separate.

## 3. Data

`driving_wp_f16`, Waymo Open Motion scenarios replayed in MetaDrive `ScenarioEnv`,
rendered as ego dashcam at $96\times96$. 40,000 train and 4,000 test clips of 16 frames
at 10 Hz, from disjoint scenarios. `annotations.npz` carries 21 keys including per-frame
`ego_position`, `ego_heading` and `ego_speed`, a 4 s future ego path, one tracked target
agent, and lead distance / TTC / collision flags.

Both splits are on the workspace at `data/driving_wp_f16/{train,test}`, byte-exact
against their `meta.json`. `/root` is destroyed on terminate, so the permanent copy
belongs on `volume://vessl-storage/yethu-drive`.

## 4. Conditions

Five, each a **MiniDiT trained from scratch** on the same schedule, differing only in
what feeds the cross-attention.

| id | condition | source | isolates |
|---|---|---|---|
| **N** | null context | the learned single token `MiniDiT` already defaults to | the floor: does conditioning do anything |
| **D** | Drive-JEPA domain, `vitl_merge_3dataset_e50.pt` | context frames | driving-domain pretraining |
| **T** | Drive-JEPA task, `..._agent_vitl_v2.ckpt` | context frames | trajectory distillation on top of D |
| **V** | Traj-VAE, `waypoint_vae_lr5x` | context latent frames | waypoint supervision on $z$'s own grid |
| **G** | V-JEPA2 ViT-H `fpc64-256` | context frames | Phantom's own encoder, the generic reference |

Training from scratch rather than fine-tuning buys the thing the previous design could
not have: **N is a genuine budget-matched floor**, so the comparison can show that
conditioning does something at all, not only that one representation beats another. It
also removes any question of a pretrained cross-attention biasing the result.

**N already exists.** `dit10m32b100e` is precisely this recipe, 100 epochs at effective
batch 32 with cosine 2e-4 and the null context, so it is condition **N** at seed 0 and
only D, T, V and G are new runs.

There is no text condition. The Waymo set ships no captions, and a templated one built
from the annotation fields would have had to be restricted to the context window to
avoid handing text more information than $r$ gets, which made it a weak stand-in for the
incumbent. Null conditioning is the honest floor instead, and stage 2 has real umT5 text
to compare against.

## 5. The conditioning path

$r$ replaces `null_context` as `MiniDiT`'s `crossattn_emb`.

**Frame window.** $r$ sees exactly the raw frames the $k$ context latent frames cover
and nothing beyond, so there is no future leakage. `MiniWan` is causal with $4\times$
temporal compression, giving raw $[0]$ at $k{=}1$ and raw $[0..4]$ at $k{=}2$. Tubelet 2
needs an even count, so the rule is *repeat the last frame when odd*: $[0,0]$ and
$[0,1,2,3,4,4]$. Training samples $k \in \{1,2\}$ per the existing
`DIT_MIN/MAX_COND_FRAMES`, so the token count varies per sample.

**Resolution.** The ViT candidates run at $256\times256$ via bicubic upsample with
ImageNet normalisation, matching Drive-JEPA's own eval path. `VJepa2Encoder` defaults to
`img_size=96`, which RoPE admits, but that is a $6\times6$ patch grid against the
$16\times16$ these checkpoints trained on, and running them $2.7\times$ off their
training resolution would confound "do these features carry value" with "do they survive
a resolution shift".

| condition | $d$ | tokens, $k{=}1$ | tokens, $k{=}2$ |
|---|---|---|---|
| D, T | 1024 | 256 | 768 |
| G | 1280 | 256 | 768 |
| V | 16 | 144 | 288 |

768 $r$ tokens against $z$'s 720 is cheap at width 256.

**Adapter.** `LayerNorm` then `Linear`$(d, 256)$. Nothing needs identity-at-init any
more, since every condition starts from random weights rather than a shared checkpoint.

**Position.** `Attention` applies RoPE only when `is_selfattn`, so cross-attention
context arrives position-free, as a bag. Correct for text, wrong for a dense
spatial-temporal grid, so a learned per-position embedding is added to the $r$ tokens
before the adapter.

## 6. Training scope

**Every condition trains from scratch**, 100 epochs at effective batch 32, matching
`dit10m32b100e`'s recipe exactly so that **N** is comparable to the rest without being
re-run at seed 0. Trainable: the whole DiT, the $r$ adapter, the positional embedding.
Frozen: the VAE and every $r$ encoder.

The alternative, fine-tuning from `dit10m32b100e`, was rejected for a specific reason.
That checkpoint's cross-attention never saw more than one key: `_run_blocks` expands
`null_context` to $(B, 1, 256)$, and softmax over a single logit is identically 1, so the
sublayer collapses to a learned per-timestep bias. Because
$\partial\,\mathrm{softmax}(s)/\partial s = 0$ at one logit, `q_proj` and `k_proj` never
received gradient and remain at initialisation. Fine-tuning would therefore have started
every condition from a pathway that does not exist, and the result would have been
partly about how fast each representation can build one.

## 7. Metrics

Both axes are reported side by side, never collapsed into one score.

**Visual**, in `conditioning/lib/dit_eval.py`: mean PSNR, SSIM and LPIPS over the predicted raw
frame range only, since the context frames are exact copies by construction, plus FVD16
over the whole clip via the SHA256-pinned I3D. Per-clip values go to `per_clip.npz`,
because section 8's comparisons are paired bootstraps.

**Trajectory**, DrivingGen's definitions (arXiv:2601.01528, ICLR 2026) ported in
`conditioning/lib/traj_metrics.py` and verified bit-exact against their code: ADE and FDE in
metres against the clip's true ego motion, DTW for path shape, heading error in degrees,
and their reference-free composite of comfort, motion and curvature. $dt = 0.1$
throughout.

Two properties decide how they are read. DTW measures path *shape*, so a translation
along the direction of travel is a time shift the warp absorbs; a condition generating
the right path at the wrong speed is caught by ADE and invisible to DTW. And the
composite is near its ceiling on real motion, so it is a guard against degenerate motion
rather than a discriminator (section 10).

**The extractor is a supervised ego-motion probe**, `conditioning/lib/pose_probe.py`, not
DrivingGen's SLAM pipeline, which over 16 frames at $96\times96$ would fall through to
its own constant-velocity recovery almost everywhere. Every clip here is labelled with
exact ego pose, and the camera is rigidly mounted over a flat road, which fixes the
ground-plane homography and makes metric scale observable. It predicts per-adjacent-pair
increments in the earlier frame's body frame and composes them as SE(2) transforms, so
**integration can start from the true pose at the conditioning cutoff** and every
measured increment comes from generated frames alone. About 1.4M params, trained from
scratch on raw frames, sharing no weights with the VAE or any candidate. FTD is deferred
to stage 2, where clip length gives more than one MTR window.

## 8. Runs and statistics

5 conditions $\times$ 3 seeds, of which N at seed 0 already exists, so **14 new runs**.

One primary metric per axis: **FVD16** for visual, **ADE** for trajectory. A condition
separates from **G** on an axis only if the gap between their 3-seed means exceeds twice
the pooled across-seed standard deviation and, on the trajectory axis, the probe's floor.
Separating on either axis carries a condition into stage 2. Primary comparisons are D
against G, T against D, V against G, and each against N, by paired bootstrap over test
clips.

## 9. Assets

| asset | state |
|---|---|
| `vae_50e24b` | ready. 50/50 epochs, `val_recon` 0.000806, LPIPS 0.0189 |
| `dit10m32b100e` | ready. **Condition N at seed 0.** `val_flow` 0.3999 |
| `waypoint_vae_lr5x` | ready. Condition **V**'s encoder. Val ADE 1.513 m over 4 s |
| data, both splits | ready, byte-exact, on the workspace |
| ego-motion probe | **trained.** Floors in section 10 |
| metric code | ported, bit-exact against DrivingGen, 37 tests green |
| Drive-JEPA **T** | on the volume, 3.5 GB |
| Drive-JEPA **D** | **absent.** Needs fetching; D against T is the domain-versus-task question |
| V-JEPA2 ViT-**H** | **absent.** Only ViT-L is on the volume. Fetch ViT-H, or redefine **G** as ViT-L and say so |
| `RSource`, adapter | **unwritten.** `R_SOURCES` registers only `traj_vae` |

## 10. Measured so far

**Baseline visual**, `dit10m32b100e/best`, all 4,000 test clips, seed 0:

| $k$ | PSNR | SSIM | LPIPS | FVD16 |
|---|---|---|---|---|
| 1 | 24.686 | 0.7062 | 0.1088 | 247.33 |
| 2 | 26.420 | 0.7546 | 0.0725 | 215.35 |

**Probe floors**, held-out test split: **0.2105 m** ADE on real frames (0.053 deg
heading), against **0.4351 m** on VAE round-trips of the same clips. The round-trip
figure is the one that binds, since generated video passes through that decoder plus DiT
error, and the 2.07x gap means probe accuracy tracks appearance quality. A
`--vae-aug` retrain is the fix.

**Trajectory calibration on real ego motion**: comfort 0.9115, curvature 0.9276 with a
median of 0.9973, motion 0.6084, composite 0.7317. 11% of clips never exceed
$v_{static}$ and 18% cover at most 1 m, so those fractions drop out of curvature and
comfort respectively.

**FVD is badly biased at small $n$**: the same configuration scores 522.77 on 64 clips
against 247.33 on 4,000. FVD16 is computed at the full 4,000 every time.

## 11. Compute

Measured: 245 s/epoch for the z-only DiT on 2 GPUs at effective batch 32, and 639 s for
a full 4,000-clip eval on one GPU. A frozen ViT-L forward over ~512 average $r$ tokens
adds about 88 s/epoch, and ViT-H about 183 s.

At 100 epochs that is 6.9 h for V, 9.3 h each for D and T, and 11.9 h for G, so
**37.4 GPU-pair-hours per seed** for the four new conditions, or about 10 h of wall on 8
GPUs with G given four ranks. Evaluation is embarrassingly parallel and costs ~15 min of
wall for the set. Three seeds is roughly 30 h plus two extra N runs.

Caching $r$ is not viable: 40,000 clips at ~512 tokens by 1024 dims in bf16 is about
42 GB per encoder against 70 GB free, with three encoders to serve.

## 12. Layout and node quirks

`conditioning/lib/` is the library, `scripts/` the entry points and node runners, `tests/` the
suite. Everything lives under `/root/cond-drivejepa-exp/` on the workspace; nothing is
written to `/root` itself or into the read-only `Task_specific_JDM` checkout.
`scripts/verify_sync.sh` proves local and node match by checksum.

The image ships a forward-compat `libcuda` (530) that the host's 525 driver cannot use,
which reads as CUDA error 804 despite `nvidia-smi` listing the GPUs. `.runtime/cudafix`
on `LD_LIBRARY_PATH` fixes it, and **`ldconfig` must not be re-run** or the fault
returns. Python block-buffers stdout over ssh, so a finished run can look silent and can
report exit 255 on a dropped connection; read the output files.

## 13. What this does not establish

A single-branch conditioning result bounds the Phantom coupling only loosely.
Conditioning on context-frame $r$ hands the DiT a semantic read of frames it already
holds as clean latents, whereas Phantom's $r$ branch predicts *future* $r$ and couples it
inside every block. A null result here says the cheap path carries nothing; it does not
prove the coupled path does.

The reference-free composite cannot carry the trajectory axis at this clip length, being
near its ceiling on real motion, so ADE, FDE and DTW do the discriminating.

The probe reads motion magnitude rather than direction: a time-reversed clip does not
reverse its sign, because dashcam footage contains no reverse motion to learn from. It is
accurate on forward driving, which is all that is generated, but it is not general visual
odometry and should not be described as such.

Nothing here is comparable to any published table. Stage 1 numbers are internally
comparable only.
