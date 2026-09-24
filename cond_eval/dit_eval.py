"""Evaluate a trained MiniDiT/PhantomDiT checkpoint's Video2World rollouts
against ground truth on a held-out test set.

A copy of ``f_toy/evaluation/dit_eval.py`` with the trajectory axis added.
``Task_specific_JDM`` is read-only for this project, so the visual half is
reproduced here rather than patched there; it imports the shared pieces
(``f_toy.evaluation.metrics``, ``f_toy.engine.dit``, ``f_toy.models.vae_loss``)
and so needs ``Task_specific_JDM`` on ``sys.path``.

**Visual axis, unchanged from the original.** Mean PSNR/SSIM/LPIPS (PSNR and
SSIM share ``f_toy.evaluation.vae_eval``'s whole-set-average convention; LPIPS
reuses ``f_toy.models.vae_loss.lpips_loss``, the same wrapper the VAE training
loop uses -- alex backbone, [0,1] inputs, time folded into batch -- and is
lower-is-better, unlike the other two) over only the *predicted*
(non-conditioned) raw frame range. The conditioned frames are exact copies of
the ground truth by construction, so including them would inflate the score. The
exception is FVD: I3D needs the whole clip, and the video-prediction
literature's FVD16 is computed over all 16 frames context included, so it is
too -- the context frames are the same ground truth for every model, so they
shift every model's FVD alike.

**Trajectory axis, new.** With a ``pose_fn`` supplied, ADE, FDE, DTW, heading
error and DrivingGen's reference-free quality composite are computed from poses
read out of the generated pixels. See ``cond_eval.traj_metrics`` for the metric
definitions and their provenance. Three points about how they are applied:

* They are computed over the **whole 16-frame** pose track, following the FVD16
  convention above rather than the PSNR one. Two reasons: frame 0 is the origin
  every pose is expressed against, and comfort and curvature differentiate the
  track twice and three times, so they need the full window to mean anything at
  T=16. ADE and FDE are additionally reported restricted to the predicted range
  as ``*_pred_range``, which is the more sensitive pair.
* With ``pose_floor`` on, ``pose_fn`` also runs on the real clip, so the probe's
  own error against the labels lands in the same ``metrics.json``. That number
  is the noise floor: differences between conditions smaller than it are not
  readable. It is a property of the probe rather than of the run, so it will
  repeat across runs sharing a probe, which is the point.
* Per-clip values for every metric are written to ``per_clip.npz`` alongside the
  summary, because the comparisons this feeds are paired bootstraps over clips
  and a mean alone cannot support one.

``pose_fn(videos) -> (xy, heading)`` takes ``(B,3,T,H,W)`` in [0,1] and returns
``xy`` of ``(B,T,2)`` in metres as (forward, lateral) in frame 0's ego body
frame, and ``heading`` of ``(B,T)`` in radians relative to frame 0, or None to
skip the heading metric. That is exactly the parameterisation
``traj_metrics.to_ego_frame`` produces from the labels, so a probe trained on its
output satisfies this by construction.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from f_toy.data import dataset_channel_order
from f_toy.engine.dit import encode_latents, stack_latent_stats
from f_toy.evaluation.metrics import (frechet_distance, i3d_features, load_i3d, psnr, ssim,
                                      to_uint8_thwc)
from f_toy.models.vae_loss import lpips_loss

from .traj_metrics import to_ego_frame, trajectory_metrics


def condition_raw_frame_cutoff(num_condition_frames, temporal_compression_factor):
    """Convert a latent-frame conditioning count to the raw (pre-VAE) frame
    index where the model's own predictions actually start: latent frame 0
    has receptive field raw frame 0 alone, and every later latent frame
    pools a ``temporal_compression_factor``-frame causal window -- see
    ``f_toy/models/vae.py``'s module docstring and
    ``scripts/compare_dit_conditioning.py``'s identical computation.
    ``num_condition_frames=0`` (unconditional) -> cutoff 0, nothing is real."""
    if num_condition_frames <= 0:
        return 0
    return 1 + (num_condition_frames - 1) * temporal_compression_factor


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _gt_pose(labels):
    """Frame-0 ego-frame pose track from a batch's label dict, or ``(None, None)``
    if the split's ``annotations.npz`` lacks the ego fields. Accepts tensors, as
    the loader yields, or arrays."""
    if "ego_position" not in labels or "ego_heading" not in labels:
        return None, None
    return to_ego_frame(_to_numpy(labels["ego_position"]), _to_numpy(labels["ego_heading"]))


def _run_pose_metrics(pose_fn, videos, gt_xy, gt_heading, cutoff):
    """``pose_fn`` on one batch of video, scored against the label track.

    The ground truth and cutoff are passed so the probe can splice: integration
    begins at the true pose at ``cutoff`` and only increments over generated
    frames contribute. A ``pose_fn`` that takes only ``videos`` still works and
    integrates from the origin instead."""
    with torch.no_grad():
        try:
            pred_xy, pred_heading = pose_fn(videos, gt_xy, gt_heading, cutoff)
        except TypeError:
            pred_xy, pred_heading = pose_fn(videos)
    pred_xy = np.asarray(pred_xy, dtype=float)
    pred_heading = None if pred_heading is None else np.asarray(pred_heading, dtype=float)
    return trajectory_metrics(pred_xy, gt_xy, pred_heading,
                              None if pred_heading is None else gt_heading,
                              pred_from=cutoff)


def run_dit_eval(sample_fn, vae, latent_stats, test_loader, dit_cfg, device, eval_dir,
                 num_condition_frames, steps, run="", which="best", dataset_dir="", log_every=20,
                 lpips_net="alex", extra=None, fvd=True, channel_order=None,
                 pose_fn=None, pose_floor=True, i3d_path=None):
    """``sample_fn(videos, shape_z, condition_latent, num_condition_frames, target_frames)
    -> (B,3,T,H,W)`` video in ``[0,1]`` -- a closure over the already-loaded
    DiT (and, for a joint model, its r-branch shape) built by the caller, so
    this function stays agnostic to plain vs. joint sampling (see
    ``f_toy.engine.dit_sample``'s ``sample``/``sample_phantom``).

    ``videos`` is the batch's raw ground truth, passed so an r-conditioned
    closure can encode r's observed slots; it re-encodes them from the context
    frames alone, so nothing past the context reaches them. Plain-DiT closures
    ignore it. The context frames are inside ``cutoff`` below and so excluded
    from the pixel scores.

    ``pose_fn``/``pose_floor``: the trajectory axis, see the module docstring.
    Without ``pose_fn`` this behaves as the original did."""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    Tz = vae.latent_num_frames(dit_cfg["FRAMES"])
    Hz = Wz = dit_cfg["IMG"] // vae.spatial_compression_factor
    if num_condition_frames >= Tz:
        raise ValueError(f"num_condition_frames={num_condition_frames} >= this VAE's {Tz} latent frames "
                         f"(FRAMES={dit_cfg['FRAMES']}) for the whole clip -- nothing would be left to "
                         f"predict. --num-condition-frames counts LATENT frames, not raw frames -- latent "
                         f"frame 0 covers raw frame 0 alone, every later one pools "
                         f"{vae.temporal_compression_factor} raw frames, so valid values here are 0..{Tz - 1}")
    mean, std = stack_latent_stats(latent_stats, Tz, device)
    cutoff = condition_raw_frame_cutoff(num_condition_frames, vae.temporal_compression_factor)

    # alex is the lightest of the three LPIPS backbones and the pip package's own
    # default, matching engine/vae.py's training-time LPIPS and eval_dit_quality.py.
    import lpips as lpips_pkg
    lpips_model = lpips_pkg.LPIPS(net=lpips_net).to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)

    i3d, real_feats, fake_feats = None, [], []
    if fvd:
        # ``i3d_path`` is explicit because f_toy's default I3D_PATH is relative to
        # the Task_specific_JDM checkout, and this project does not write there.
        # The SHA256 pin inside load_i3d applies either way.
        i3d = load_i3d(device, i3d_path)
        channel_order = channel_order or dataset_channel_order(dit_cfg.get("DATASET", "cubetoy"))

    psnr_values, ssim_values, lpips_values = [], [], []
    traj_batches, floor_batches = [], []
    n_total = 0
    t0 = time.time()
    for step, (videos, labels) in enumerate(test_loader):
        videos = videos.to(device)
        B = videos.shape[0]
        condition_latent = encode_latents(vae, videos, mean, std)
        shape_z = (B, dit_cfg["VAE_ZDIM"], Tz, Hz, Wz)
        pred = sample_fn(videos, shape_z, condition_latent, num_condition_frames, dit_cfg["FRAMES"])
        if i3d is not None:
            # whole clip, context included -- FVD16's convention (see module docstring)
            real_feats.append(i3d_features(i3d, videos, channel_order))
            fake_feats.append(i3d_features(i3d, pred, channel_order))

        # Trajectory, before the cutoff slice: pose is expressed relative to frame 0,
        # which the slice would remove. See the module docstring.
        if pose_fn is not None:
            gt_xy, gt_heading = _gt_pose(labels)
            if gt_xy is None:
                raise ValueError("pose_fn was given but this split's annotations carry no "
                                 "ego_position/ego_heading, so there is nothing to score against")
            traj_batches.append(_run_pose_metrics(pose_fn, pred, gt_xy, gt_heading, cutoff))
            if pose_floor:
                floor_batches.append(_run_pose_metrics(pose_fn, videos, gt_xy, gt_heading, cutoff))

        pred, gt = pred[:, :, cutoff:], videos[:, :, cutoff:]

        for o, r in zip(to_uint8_thwc(gt), to_uint8_thwc(pred)):
            psnr_values.append(psnr(o, r))
            ssim_values.append(ssim(o, r))
        # Per sample, like PSNR/SSIM above, so all three average the same way.
        # Scored on the float tensors rather than the uint8 round-trip: LPIPS is
        # defined on continuous [0,1] input, and clamping matches to_uint8_thwc's.
        with torch.no_grad():
            for i in range(B):
                lpips_values.append(float(lpips_loss(lpips_model, pred[i:i + 1].clamp(0, 1),
                                                     gt[i:i + 1].clamp(0, 1))))
        n_total += B

        if (step + 1) % log_every == 0:
            print(f"  {n_total}/{len(test_loader.dataset)}  ({time.time() - t0:.0f}s)")

    n = len(psnr_values)
    mean_psnr, mean_ssim = float(np.mean(psnr_values)), float(np.mean(ssim_values))
    mean_lpips = float(np.mean(lpips_values))
    fvd_value = (frechet_distance(np.concatenate(real_feats), np.concatenate(fake_feats))
                 if i3d is not None else None)
    fvd_str = f" FVD={fvd_value:.2f}" if fvd_value is not None else ""

    # nanmean, not mean: comfort and curvature are NaN for a track that never moved
    # (traj_metrics preserves upstream's convention), and a stationary clip must not
    # poison the whole column.
    per_clip = {k: np.concatenate([b[k] for b in traj_batches]) for k in traj_batches[0]} \
        if traj_batches else {}
    traj_means = {k: float(np.nanmean(v)) for k, v in per_clip.items()}
    per_clip_floor = {k: np.concatenate([b[k] for b in floor_batches]) for k in floor_batches[0]} \
        if floor_batches else {}
    floor_means = {f"floor_{k}": float(np.nanmean(v)) for k, v in per_clip_floor.items()}

    traj_str = ""
    if traj_means:
        traj_str = (f" ADE={traj_means['ade']:.4f} FDE={traj_means['fde']:.4f} "
                    f"DTW={traj_means['dtw']:.3f} Q={traj_means['quality_composite']:.4f}")
        if "heading_error_deg" in traj_means:
            traj_str += f" HEAD={traj_means['heading_error_deg']:.3f}deg"
        if floor_means:
            traj_str += f" (probe floor ADE={floor_means['floor_ade']:.4f})"
    elapsed = time.time() - t0

    # ``extra``: caller-supplied run identity that belongs in the record, e.g. which
    # r-conditioning condition the caller ran under.
    extra = dict(extra or {})
    extra_str = "".join(f" {k}={v}" for k, v in extra.items())
    summary = (f"[eval_dit] run={run} which={which} dataset={dataset_dir} n={n} "
               f"num_condition_frames={num_condition_frames} steps={steps}{extra_str} "
               f"PSNR={mean_psnr:.3f} SSIM={mean_ssim:.4f} LPIPS={mean_lpips:.4f}"
               f"{fvd_str}{traj_str} ({elapsed:.0f}s)")
    print(summary)
    (eval_dir / "eval.log").write_text(summary + "\n")

    metrics = dict(run=run, which=which, dataset_dir=str(dataset_dir), num_samples=n,
                   num_condition_frames=num_condition_frames, steps=steps,
                   predicted_from_raw_frame=cutoff,
                   psnr=mean_psnr, ssim=mean_ssim, lpips=mean_lpips, lpips_net=lpips_net,
                   fvd=fvd_value, fvd_channel_order=channel_order if fvd else None,
                   **traj_means, **floor_means,
                   **extra, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {eval_dir / 'metrics.json'}")

    if per_clip:
        np.savez(eval_dir / "per_clip.npz",
                 psnr=np.array(psnr_values), ssim=np.array(ssim_values),
                 lpips=np.array(lpips_values), **per_clip,
                 **{f"floor_{k}": v for k, v in per_clip_floor.items()})
        print(f"saved {eval_dir / 'per_clip.npz'}")
    return metrics
