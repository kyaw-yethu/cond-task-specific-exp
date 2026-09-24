# conditioning

Training and evaluation of the conditioned MiniDiT, stage 1 and stage 2. Each condition
feeds a different representation into MiniDiT's cross-attention and is scored on video
quality (FVD, PSNR, SSIM, LPIPS) and on the ego trajectory read back from the generated
frames. Needs `third_party/Task_specific_JDM` (`f_toy`) on the path.

| tag | conditioning |
|---|---|
| N | none |
| D | Drive-JEPA, domain-adapted encoder |
| T | Drive-JEPA, task encoder |
| V | Traj-VAE latent |
| G | V-JEPA2 ViT-H |

## lib

| module | |
|---|---|
| `cond_dit.py` | `ConditionedDiT`, cross-attention conditioner |
| `r_sources.py` | conditioning sources: context frames to tokens, per tag |
| `cached.py` | fixed-k VAE latents and tokens as memmaps |
| `dit_eval.py` | video and trajectory metrics for one checkpoint |
| `traj_metrics.py` | DrivingGen trajectory metrics, ported |
| `pose_probe.py` | ego-motion probe, the stage-1 trajectory extractor |
| `traj_v2.py` | Traj-VAE v2 |

## scripts

**Stage 1**: `build_cache.py`, `train_dit_cond.py`, `eval_cond.py`, `make_results.py`,
`make_comparison.py`, `make_samples.py`, `train_probe.py`, `calibrate_traj.py`,
`convert_drivejepa.py`, `eval_dit.py`, `diag_per_frame.py`.

**Stage 2**: `prep_traj_targets.py`, `train_traj_v2.py`, `train_vae_ft.py`,
`check_wan_vae.py`, `show_vae_recon.py`.

**Figures**: `plot_*.py`, sharing `figstyle.py`.

**Node drivers**: `node_build_and_train.sh` (cache, then train one tag),
`node_build_cache.sh`, `node_train_cond.sh`, `node_eval_cond.sh`, `node_train_probe.sh`,
`node_orchestrate.sh` (evaluate, compare and write results once training ends),
`node_orchestrate_k1.sh` with `node_k1_chain.sh` (the k=1 `dit_large` screen),
`node_check_figs.sh`, `setup_cudafix.sh`.

```bash
bash conditioning/scripts/node_train_probe.sh
bash conditioning/scripts/node_build_and_train.sh 0 N none
bash conditioning/scripts/node_build_and_train.sh 1 T drivejepa drivejepa_T_vitl256
GPU=5 bash conditioning/scripts/node_eval_cond.sh T
python conditioning/scripts/make_results.py --out-root out --proj .
```

`configs/` holds the DiT configs, `tests/` the pytest suite, `docs/` the stage plans,
the probe write-up and results.
`runs/` records the MiniWan, probe and Traj-VAE v2 training runs: where the
checkpoints are, the settings, the results and the logs.
