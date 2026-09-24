# Stage 2: driving representation as conditioning, at Phantom's scale

**Question, unchanged from stage 1.** Replace a video DiT's text conditioning with a
driving-specific representation. Does it buy visual quality, trajectory quality, or
neither?

**What changes.** Wan2.2-TI2V-5B in place of `MiniDiT`, nuScenes CAM_FRONT in place of
rendered Waymo, $256\times448$ in place of $96\times96$, and DrivingGen's own extractor
in place of a supervised probe.

Still the **video branch (of Phantom) only**.


## 1. The backbone

| | |
|---|---|
| model | **Wan2.2-TI2V-5B**, Phantom's own choice |
| VAE | $16\times16\times4$, reaching $4\times32\times32$ with patchification |
| text encoder | umT5-xxl, the pathway $r$ replaces |
| native resolution | $1280\times704$ / $704\times1280$, 720P at 24 fps |
| Phantom's training setting | $480\times832$, up to 121 frames, visual branch frozen, physics branch and dual cross-attention trained, 4x H200, global batch 128, 2 epochs, ~400K video-text pairs |

Token count at $256\times448$ over 69 frames: $T_z = 1 + (69-1)/4 = 18$,
$H = 256/32 = 8$, $W = 448/32 = 14$, so $18 \cdot 8 \cdot 14 = 2{,}016$ tokens, of
which each latent frame is 112. 

Phantom's own setting, $480\times832$ over 121 frames,
is $31 \cdot 15 \cdot 26 = 12{,}090$, six times more.

**The one thing that inverts relative to stage 1.** `MiniDiT`'s cross-attention never
saw more than one key, so its $W_q$ and $W_k$ were still at initialisation and a full
fine-tune was *forced*. Wan2.2 trained its cross-attention against hundreds of
thousands of distinct umT5 sequences, so the pathway here is real and selective.
Training the $r$ adapter plus each block's cross-attention $K/V$, with everything else
frozen, is therefore both the scientifically cleaner choice and, on 24 GB cards, the
only feasible one. It also matches Phantom, which freezes every pretrained
visual-branch parameter.

## 2. Data

Everything lives on the VESSL storage volume **`volume://vessl-storage/yethu-drive`**,
under `assets/nuscenes/`: **37.17 GB across 11,898 entries**. Split as in `trd-drivejepa-exp/docs/DATA_INVENTORY.md`: 600
train scenes, 250 probe-pool scenes.

| subtree | contents |
|---|---|
| `cam_front_shards/` | **850 tars**, `cam_front_scene-0001` … `-1110`, one uncompressed tar per scene holding that scene's CAM_FRONT jpgs at native $900\times1600$, 12 Hz, 198,315 frames, 4.6 h. Members are not in temporal order; sort by the microsecond timestamp ending each filename |
| `cam_front_index/` | `_manifest.json` (49 KB), `scene_files.json` (15.7 MB) |
| `raw/v1.0-trainval/` | all 13 metadata tables, including **`ego_pose.json` (615.8 MB)**, **`calibrated_sensor.json` (3.1 MB)**, `sample_data.json` (1.3 GB), `sample_annotation.json` (556.4 MB), `sample.json`, `scene.json`, `instance.json`, `category.json`, `attribute.json`, `visibility.json`, `sensor.json`, `log.json`, `map.json` |
| `raw/can_bus/` | **980 `_pose.json`** plus `ms_imu`, `steeranglefeedback`, `vehicle_monitor`, `zoe_veh_info`, `zoesensors`, `route`, `meta` per scene |

| | |
|---|---|
| resolution | $256\times448$, from a $1575\times900$ centre crop |
| frame rate | uniform 10 Hz, $dt = 0.1$ |
| clip length | $F = 69$, $T_z = 18$, **6.9 s** |
| clips per scene | 3, tiling, ~8% overlap |
| train | 600 scenes, **1,800 clips** |
| eval | 250 scenes, **750 clips** |
| tokens per clip | **2,016** |

### Resolution

nuScenes CAM_FRONT is $1600\times900$, exactly 16:9. Wan2.2 patchifies $16\times$ in the
VAE and $2\times$ in the patch embedding, so both spatial dimensions must be multiples
of 32. Writing $H = 32a$ and $W = 32b$, an exact 16:9 grid needs $a/b = 9/16$, whose
smallest solution is $288\times512$. Anything below that needs a centre crop first.

**$256\times448$** takes $a = 8$, $b = 14$, an aspect of $4/7$, reached by cropping
$1600\times900$ to $1575\times900$ and resizing.

### Frame rate

Native CAM_FRONT timing averages exactly 12.00 Hz but is **not uniform**: inter-frame
gaps alternate between 100 ms and 50 ms in roughly a 2:1 ratio. Two consequences follow.
Training on natively spaced frames teaches the model motion that stutters by a factor
of two, and no single $dt$ is correct for the derivative-based trajectory metrics.

Snapping each scene to a uniform 100 ms grid removes both. It costs 16% of the frames,
198,315 down to **166,743**, and yields **$dt = 0.1$ exactly**, which is DrivingGen's
own default and the grid `DG_TRAJECTORY_FLOOR.md` measures on.

### Clip length and count

850 scenes hold a median of 234 native frames over 19.5 s, minimum 190. On the 10 Hz
grid a scene holds a median of 196 frames, minimum 158.

With $F = 69$ and three clips per scene placed at $0$, $(N-F)/2$ and $N-F$, the clips
**tile each scene exactly**: every frame is used, with a median overlap of 5 frames at
each junction, 8% of a clip, and 11% at the 95th percentile. All 850 scenes support
three clips, including the 158-frame outlier. $F \equiv 1 \pmod 4$ is required for the
causal VAE to give an integer latent length, $T_z = 1 + (F-1)/4 = 18$.



> **Data volume is the binding constraint and it is severe.** 1,800 training clips against
Phantom's ~400K video-text pairs is 222x fewer, and the honest independence count is
lower still: **600 scenes**, since three clips from one 20 s drive share weather,
location, time of day and vehicle. 

## 3. Conditions

Six, as in stage 1, each differing only in what feeds the cross-attention adapter.

| id | condition | source | isolates |
|---|---|---|---|
| **N** | umT5 with an empty prompt | the backbone's own null pathway | floor: does conditioning do anything |
| **X** | umT5 with a templated caption | nuScenes annotations and CAN | the incumbent being replaced |
| **G** | V-JEPA2 ViT-H `fpc64-256` | conditioning window | **Phantom's own encoder**, the generic reference |
| **D** | Drive-JEPA domain | conditioning window | driving-domain pretraining |
| **T** | Drive-JEPA task | conditioning window | trajectory distillation on top of D |
| **V** | Traj-VAE on Wan2.2 latents | conditioning window | waypoint supervision on $z$'s own grid |


> **V does not exist yet and is a prerequisite, not a condition.** Traj-VAE is a clone of
a VAE encoder fine-tuned to forecast ego waypoints. Stage 1's lives on `MiniWan`'s
$16\times5\times12\times12$ grid. A Wan2.2 equivalent has to be trained first: clone the
Wan2.2 VAE encoder, freeze its decoder, attach the Drive-JEPA-style query decoder head,
and supervise on nuScenes ego waypoints. Until that exists, **V is out** and the
comparison is five conditions.

## 4. The conditioning window

**The backbone conditions per latent frame, not on a single image.**
`WanModel.forward` expands the timestep with `t.expand(t.size(0), seq_len)` and the
AdaLN modulation leaves `time_projection` shaped $(B, T, 6, d_{\text{model}})$, so every
latent frame carries its own noise level and its own modulation. TI2V-5B's
image conditioning is latent replacement on top of that: the conditioning frame's
noised latent is overwritten with its clean latent before the blocks, the loss on that
frame is masked, and the clean latent is re-injected at every sampling step. There is no
conditioning channel and no separate mask input, so $\text{in\_dim} = \text{out\_dim}$.

Multi-frame conditioning therefore needs no change to the architecture. Set $t = 0$ on
the first $k$ latent frames, replace their latents with the clean ones, and mask the
loss there. This is the same Video2World scheme stage 1 used, expressed through the
per-frame timestep rather than through an extra mask channel.

**Conditioning lengths.** The VAE is causal at temporal stride 4, so latent frame 0
covers raw frame 0 and every later one pools 4: $k$ latent frames cover
$n = 1 + 4(k-1)$ raw frames. Training samples

$$k \in \{1,\ 2,\ 5,\ 7\} \quad\Longrightarrow\quad n \in \{1,\ 5,\ 17,\ 25\}$$

uniformly per clip, matching Phantom's exposure to varying temporal context.

**The schedule is scaled to $T_z = 18$.** At $T_z = 31$ the four values $\{1, 4, 8, 12\}$
spend on average 20.2% of the sequence on frames carrying no loss. Carrying those same
values to $T_z = 18$ would spend 34.7%, and $k = 12$ alone would leave 6 latent frames
to generate against 45 raw frames held clean. Scaling by $18/31$ gives
$\{1, 2, 5, 7\}$, which restores the 20.2% property at 20.9% and keeps the longest
context at 25 of 69 raw frames.


**$r$ encodes exactly the $n$ conditioning frames**, the same frames the DiT receives as
clean latents. Two properties follow, and they are the reason for the choice:

- **No leakage of any kind.** Every frame $r$ sees is already held clean by the DiT, so
  $r$ can carry nothing from the generated range. A window extending before the clip
  would also avoid future leakage, but it would hand the conditioned runs more real
  frames than **N** receives.
- **The comparison is controlled.** $r$ contributes no additional pixels, only a
  semantic read of pixels the DiT already has. Any difference between **N**, **X**,
  **G**, **D** and **T** is therefore attributable to the representation rather than to
  observation length.

**Conditioning frames cost sequence budget.** Each latent frame is $8 \times 14 = 112$ of the 2,016 tokens, and those frames are not supervised:

| $k$ | $n$ | latent frames generated | raw frames generated | tokens spent on conditioning |
|---|---|---|---|---|
| 1 | 1 | 17 | 68 | 5.6% |
| 2 | 5 | 16 | 64 | 11.1% |
| 5 | 17 | 13 | 52 | 27.8% |
| 7 | 25 | 11 | 44 | 38.9% |

Averaged over the four, 20.9% of every step is spent on frames carrying no loss, and
even the longest context still generates 4.4 s of the 6.9 s clip.

**Adapter.** `LayerNorm` $\rightarrow$ `Linear`$(d, d_{\text{model}})$, initialised so
that at step 0 the adapter's output matches the empty-prompt umT5 embedding, so every
condition starts bit-exact at **N**. Same identity-at-init discipline as stage 1. The
learned positional embedding is sized for the largest window and sliced per $k$.

**Encoder resolution.** The ViT candidates run at $256\times256$, their own training
resolution, from a resize of the $256\times448$ window. A tubelet-2 encoder needs an even
frame count, so $n$ is padded by repeating the last frame, giving
$\lceil n/2 \rceil \times 256$ tokens.

**Caching.** $r$ tracks $k$, so each clip needs one cached entry per value of $k$:
$1 + 3 + 9 + 13 = 26$ tubelets, so 6,656 tokens per clip across all four.

| encoder | per clip | 2,550 clips |
|---|---|---|
| **D**, **T** ViT-L, $d = 1024$ | 13.6 MB | 34.8 GB each |
| **G** ViT-H, $d = 1280$ | 17.0 MB | 43.4 GB |
| Wan2.2 VAE latents, 48 ch | 0.77 MB | 2.0 GB |

About 115 GB for all three encoders plus latents, against 357 GB free on the node's
`/dev/shm`. Caching is what keeps the encoders and the VAE out of GPU memory during
training, which §6's budget depends on.

**One risk to retire before building.** The checkpoint has only ever seen $k = 1$, and
the machinery that would exploit several clean frames is self-attention and the
per-frame modulation, neither of which is in a trainable set restricted to
cross-attention $K/V$. Run the pretrained checkpoint at $k = 5$ and $k = 7$ and check whether
the continuation respects the extra frames. If it does not, add `time_projection` and
the per-block modulation parameters to the trainable set, which is 57M against 566M.

## 5. Evaluation

**Visual.** FVD at $256\times448$, plus PSNR/SSIM/LPIPS over the generated frame range.
FVD must be computed at the full clip count every time: stage 1 measured the same
configuration at 522.77 on 64 clips and 247.33 on 4,000, a factor of 2.1 of
small-sample bias, so no subset figure is comparable to a full one.

**Trajectory.** DrivingGen's pipeline, which at 6.9 s on real front-view RGB is inside
its design envelope rather than outside it as at stage 1. FTD also becomes available:
its MTR `agent_polyline_encoder` windows are $H{=}10$ at stride 10, giving 6 windows at
69 frames, against the single degenerate window a 16-frame clip allowed. The measured
floor of this pipeline on real nuScenes footage is in `DG_TRAJECTORY_FLOOR.md`; it was
taken at $1024\times576$ over 101 frames and has to be re-measured at $256\times448$
over 69 frames before any condition is compared.

Four specifics from their code that have to be handled:

1. **They predict intrinsics rather than reading calibration.**
   `extract_traj_ego_unidepth.py:297` takes `intrinsics` from UniDepthV2's own output;
   the branch that loads them from a file is present but commented out. Correct for
   their open-domain track, wasteful here. `calibrated_sensor.json` is on the volume
   (§7), so re-enable that branch and feed real intrinsics.
2. **Metric scale comes entirely from UniDepthV2**, and their alignment runs
   `with_scale=False`, so nothing is ever rescaled to the reference. ADE is therefore
   directly exposed to UniDepth's absolute-scale error on generated frames. SLAM does
   not avoid the learned-model-on-degraded-input problem; it relocates it into a depth
   network.
3. **The failure-recovery path can eat the measurement.** Failed pose estimation is
   replaced by constant-velocity extrapolation with random orientation perturbations.
   Degraded generated video triggers it more often than real video, so a condition whose
   output cannot be reconstructed gets scored by the fallback instead of penalised. The
   per-condition failure rate is logged beside ADE or the number is uninterpretable.
4. **$dt = 0.1$**, per §2, which is DrivingGen's default and needs no override.

**A supervised probe as the power instrument, alongside.** nuScenes ships `ego_pose`
and the probe pool has 100% CAN coverage, so the same per-step regressor stage 1
specifies is available here: predict per-adjacent-pair $(\Delta\text{fwd},
\Delta\text{lat}, \Delta\theta)$ in the previous frame's body frame, compose the SE(2)
transforms, and **integrate from the true pose at the conditioning boundary** so every
measured increment comes from generated frames alone. Cheaper than ORB-SLAM2 plus
UniDepthV2 per clip per condition, and with a tighter floor. Its numbers are internally
comparable only, which is why both run.

**Calibrating the benchmark itself.** Because nuScenes has ground-truth ego pose,
DrivingGen's extractor can be run on *real* clips and its own error measured. That is
the noise floor of the benchmark's trajectory metrics and it appears to be unpublished:
if their SLAM lands at some ADE $\epsilon$ on real video, every difference below
$\epsilon$ in their table of 14 models is noise. It is cheap, it stands as a result on
its own, and it tells us how to read our own numbers. Do it first.

**Cross-check.** Agreement between SLAM and probe on held-out real clips means either
can be trusted; disagreement localises which fails on generated input.

## 6. Compute, the largest risk

Phantom used 4x H200 at 141 GB. The VESSL workspace `yt-workspace` is **4x RTX 3090 at
24 GB**, and the model card's 24 GB figure is for **inference with offloading**, not
training.

At 5B parameters, bf16 weights alone are ~10 GB. With the trunk frozen there is no
optimizer state for the bulk, and the trainable set is the adapter plus cross-attention
$K/V$, so the question is entirely activation memory at 2,016 tokens with gradient
checkpointing, at batch 1 per card.

**A memory probe comes before anything else is built.** One forward and backward pass
at 2,016 tokens on a single 3090, with the trunk frozen and checkpointing on, decides
the whole configuration: what per-card batch is reachable, whether FSDP or ZeRO-3
sharding of the 10 GB is needed, and what gradient accumulation is required to approach
a useful effective batch. It also returns the measured seconds per step the budget
below is derived from. If it does not fit, the fallbacks in order are trimming §4's
schedule, then $224\times384$, then accepting a smaller trainable set.

**Budget.** Forward cost per clip is $2 N T_{\text{tok}} + 4 L T_{\text{tok}}^2 d
\approx 21.7$ TFLOPs at $N = 5$B, $T_{\text{tok}} = 2{,}016$, $L \approx 30$,
$d \approx 3072$. A training step is about three times that: backward through
activations reaches the $K/V$ of block 0 even with the trunk frozen, and checkpointing
adds a recompute pass. At an assumed 25 TFLOPS achieved in training and 30 in sampling
across the 4 cards:

| | |
|---|---|
| epoch, 1,800 clips | 19.5 min |
| train, 30 epochs | 9.8 h |
| eval, 750 clips at 50-step Euler | 1.9 h |
| extractor and probe | ~2 h |
| **per run** | **~13.7 h** |
| **5 conditions, 1 seed** | **~2.9 d** |


## 7. Assets

All paths below are under `volume://vessl-storage/yethu-drive/assets/`.

**On the volume**

| asset | path | state |
|---|---|---|
| nuScenes CAM_FRONT shards | `nuscenes/cam_front_shards/` | **ready.** 850 tars, 198,315 frames |
| nuScenes metadata | `nuscenes/raw/v1.0-trainval/` | **ready.** all 13 tables, `ego_pose` and `calibrated_sensor` included |
| nuScenes CAN bus | `nuscenes/raw/can_bus/` | **ready.** 980 `_pose.json`. 100% coverage on the 250-scene probe pool; 15 train scenes lack CAN |
| Drive-JEPA **task** (T) | `drive-jepa/drive_jepa_perception_based_agent_vitl_v2.ckpt` | **ready.** 3.5 GB |
| V-JEPA2 ViT-**L** | `vjepa2-vitl/` | ready, but see the gap below |
| DrivingGen source | `repos/DrivingGen/` | **ready**, including `third_parties/MTR` |
| UniDepth, YOLOv10 source | `repos/UniDepth/`, `repos/yolov10/` | **ready** |
| MTR checkpoint for FTD | `DrivingGen/mtr-epoch=28-step=176552.ckpt` | **ready** |
| DrivingGen benchmark data | `DrivingGen/{videos-fvd,ego_condition,open_domain}` | **ready.** 22,200 / 888 / 444 entries |
| DINOv3 ViT-L/16 | `dinov3-vitl16/` | **ready.** DrivingGen video-consistency metric |
| SAM 2.1 hiera-large | `sam2.1-hiera-large/` | **ready.** agent-appearance metric |
| Cosmos-Reason1-7B | `Cosmos-Reason1-7B/` | **ready.** abnormal-disappearance metric |

**Gaps, in the order they block work**

| asset | why it matters | note |
|---|---|---|
| **Wan2.2-TI2V-5B** | the backbone; nothing runs without it | only `Wan2.1-T2V-1.3B` is on the volume |
| **Drive-JEPA domain**, `vitl_merge_3dataset_e50.pt` | condition **D**, and D against T is where the domain-versus-task question lives | absent; only the task checkpoint is there |
| **V-JEPA2 ViT-H `fpc64-256`** | condition **G**, Phantom's own encoder | only ViT-**L** is on the volume. Either fetch ViT-H to match Phantom, or redefine **G** as ViT-L and say so, which costs comparability with Phantom's numbers |
| ORB-SLAM2 | DrivingGen's extractor | source is vendored; needs a C++ build on the node |
| $256\times448$ clip build | training input | to build from the shards per §2 |
| trajectory floor at the training geometry | §5 reads every condition against it | re-run the calibration at $256\times448$ over 69 frames |
| Traj-VAE on Wan2.2 latents | condition **V** | does not exist, see §3 |
| ego-motion probe | the power instrument of §5 | to train on `ego_pose` |

The three encoder and backbone gaps are all single HuggingFace fetches. The **G** gap is
the one with a design consequence rather than a download: swapping ViT-H for ViT-L
changes what "Phantom's own encoder" means in the comparison.

## 8. What this does not establish

A single-branch conditioning result bounds the Phantom coupling only loosely.
Conditioning on past-window $r$ hands the DiT a semantic read of frames it does not
otherwise see, whereas Phantom's $r$ branch predicts *future* $r$ and couples it inside
every block. A null result here says the cheap path carries nothing; it does not prove
the coupled path carries nothing.

With 1,800 training clips from 600 independent scenes against Phantom's 400K pairs, this
measures what a driving representation buys **in a low-data fine-tune of a pretrained
backbone**. It does not measure what it would buy at Phantom's data scale, and a null
result is confounded with data volume until that is stated alongside it.

$256\times448$ is a sixth of Phantom's token count and well below Wan2.2's native
resolution. Absolute FVD from this setup is internally comparable only, and the
trajectory metrics carry whatever error §5's re-calibration measures at that geometry.
The comparison between conditions is what this stage establishes; no number in it is a
published-table entry.
