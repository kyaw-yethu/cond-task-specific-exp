# Runs

Training records for the three checkpoints the experiments build on. Weights are not in
the repository. All three are in
[Google Drive](https://drive.google.com/drive/folders/1kv8P2oRp-c6zdsQd_aaVNAz1QQPCfkAx?usp=sharing)
as `miniwan/best.pt`, `probe/probe.pt` and `traj_vae/best.pt`, with the md5s below; the
originals are on the yt-workspace node, and the two WOMD runs are also backed up on the
volume.

| run | checkpoint on the node | volume backup | md5 of the checkpoint used |
|---|---|---|---|
| MiniWan | `/root/cond-drivejepa-exp/checkpoints/womd/vae_miniwan_womd/best.pt` | `yethu-drive/backup/vae_miniwan_womd.tar.gz` | `b7640cce453f83a9dcf06168da1e72e9` |
| probe | `/root/cond-drivejepa-exp/out/probe_vaeaug/probe.pt` | none | `9a81d11fb509e90e78c0100bab0a5130` |
| Traj-VAE v2 | `/root/cond-drivejepa-exp/checkpoints/womd/traj_v2/best.pt` | `yethu-drive/backup/traj_v2.tar.gz` | `226f7f76b801c5ce38852b9437aff87b` |

Each checkpoint folder also holds `last.pt`. The MiniWan and Traj-VAE checkpoints carry
their config in the file. Scripts expect them under this repository's `checkpoints/`
and `out/`, so copy or symlink them there.

## MiniWan

Causal video VAE trained from scratch on `womd_7hz_f33` (33 frames, $96\times96$,
7 Hz). 5.60M parameters, latent width 16.

```bash
torchrun --nproc_per_node 4 conditioning/scripts/train_vae_ft.py --model miniwan \
    --run-name vae_miniwan_womd --epochs 50 --batch 6 --lr 2e-4 --warmup-frac 0.1
```

| setting | value |
|---|---|
| GPUs, batch | 4, 6 per rank, effective 24; 1,666 steps/epoch, 83,300 total |
| schedule | 50 epochs, lr $2\times10^{-4}$ to $10^{-6}$ cosine, 10% warmup, wd $10^{-5}$ |
| loss | reconstruction + LPIPS (AlexNet) $\times 0.05$ + KL $\times 3\times10^{-4}$ (10% warmup) |
| other | dropout 0.1, grad clip 1.0, bf16, seed 0, input normalised by train pixel stats |
| wall time | about 820 s/epoch, 11.4 h |

Best is the last epoch (50). On all 4,000 val clips: PSNR 31.33 dB, LPIPS 0.0198, MSE
0.00081. Latent std from 4,000 train clips: frame 0 0.2818, other frames 0.3026.

Files: `miniwan/train.log`, `miniwan/history.json` (per-epoch train and val),
`miniwan/val_full.json`.

## Probe

Ego-motion probe that reads the ego trajectory out of a 16-frame $96\times96$ Waymo clip
(`driving_wp_f16`); it is the stage-1 trajectory extractor. 1.42M parameters. Trained
with VAE round-trip augmentation, which is the version used for evaluation; design and
floors in `../docs/PROBE.md`.

```bash
bash conditioning/scripts/node_train_probe.sh --vae-aug 0.5 --out-dir out/probe_vaeaug
```

| setting | value |
|---|---|
| data | 38,000 fit / 2,000 held out from train / 4,000 test |
| schedule | 20 epochs, batch 64, one GPU |
| augmentation | round trip through the Waymo VAE `vae_50e24b` with $p=0.5$ |
| wall time | 8,906 s, 2.5 h |

Final val: ADE 0.222 m, FDE 0.419 m, heading 0.093°.

| floor on the test split | ADE (m) | FDE (m) | heading (°) | p95 ADE (m) |
|---|---|---|---|---|
| real frames | 0.2247 | 0.4260 | 0.078 | 0.8385 |
| VAE round trip | 0.2739 | 0.5134 | 0.112 | 0.9267 |

Sanity controls: a repeated single frame gives 0.29 m net displacement; time reversal gives
+23.78 m against +11.76 m forwards, so the sign does not flip.

Files: `probe/train.log`, `probe/probe_floors.json`.

## Traj-VAE v2

Trajectory VAE on `womd_7hz_f33`. The encoder is cloned from the MiniWan checkpoint above
and fine-tuned, with motion, waypoint, occupancy and scene heads on top. The number of
context frames $k$ is drawn from 1 to 3 per batch. The design is in `../docs/TRAJ_VAE_PLAN.md`.
Targets come from `conditioning/scripts/prep_traj_targets.py`.

```bash
python conditioning/scripts/prep_traj_targets.py --split train
torchrun --nproc_per_node 4 conditioning/scripts/train_traj_v2.py --run-name traj_v2 --epochs 30
```

| setting | value |
|---|---|
| model | encoder 2.39M, heads 0.26M, 28 future steps |
| GPUs, batch | 4, 6 per rank, effective 24; 1,666 steps/epoch, 49,980 total |
| schedule | 30 epochs, encoder lr $5\times10^{-5}$, head lr $10^{-3}$, end at 1% of peak, 5% warmup, wd $10^{-5}$ |
| loss weights | motion 1.0, waypoint 1.0, occupancy 0.5, scene 0.25, KL $10^{-4}$ |
| other | dropout 0.1, grad clip 1.0, 500 val clips, seed 0 |
| wall time | about 340 s/epoch, 2.8 h |

Best epoch 19: val loss 1.5723, waypoint ADE 3.606 / 2.790 / 2.830 at $k$ = 1 / 2 / 3.
The last epoch (30) has a higher val loss, 2.1307.

Files: `traj_vae/train.log`, `traj_vae/history.json`.
