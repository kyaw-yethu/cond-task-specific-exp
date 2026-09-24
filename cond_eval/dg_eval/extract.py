"""DrivingGen's ego-trajectory extractor, driven over nuScenes clips.

Their `extract_traj_ego_unidepth.py` is a script, not a library: it hard-codes a
576x1024 frame, a checkpoint path from the authors' own filesystem, and drops
into `pdb` when trajectory estimation raises. So the driver loop is rewritten
here while every piece of the actual algorithm -- SIFT extraction, FLANN
matching, the distance filter, PnP-RANSAC motion estimation and the pose
composition including its failure-recovery branch -- is imported from their
`visual_slam` package and called unmodified.

Three things this adds that their script cannot report:

**The failure rate.** `estimate_motion` returns an `ok` flag that
`estimate_trajectory` consumes and never surfaces. When it is false the step is
replaced by the previous step length at a uniformly random yaw within 90
degrees, so a clip whose frames cannot be reconstructed is scored by that
fallback rather than penalised. Wrapping `estimate_motion` records the flag per
frame pair while leaving their code byte-identical.

**Intrinsics provenance.** Their line 297 takes intrinsics from UniDepth's own
prediction; the branch that loads real calibration is present but commented out.
nuScenes ships exact CAM_FRONT calibration, so all three variants can be
measured: predicted, real intrinsics handed to the SLAM alone (their commented
branch), and real intrinsics also conditioning the depth network.

**Determinism.** The failure branch draws from `np.random`, so the seed is fixed
per clip and recorded.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image


def _add_drivinggen_to_path(repo: str) -> None:
    pkg = os.path.join(repo, "drivinggen")
    for p in (pkg, os.path.join(pkg, "func")):
        if p not in sys.path:
            sys.path.insert(0, p)


class Extractor:
    """Holds the two networks so a worker loads them once for many clips."""

    def __init__(self, repo: str, unidepth_dir: str, yolo_ckpt: str,
                 device: str = "cuda", resolution_level: int = 0,
                 roi: str = "top"):
        _add_drivinggen_to_path(repo)
        import torch
        from unidepth.models import UniDepthV2
        from ultralytics import YOLOv10

        self.torch = torch
        self.depth_model = UniDepthV2.from_pretrained(unidepth_dir)
        self.depth_model.resolution_level = resolution_level
        self.depth_model.interpolation_mode = "bilinear"
        self.depth_model = self.depth_model.to(device).eval()
        self.det_model = YOLOv10(yolo_ckpt).to(device)
        self.device = device
        self.roi = roi

        from visual_slam import vo as _vo
        from visual_slam.dataset import DatasetHandler
        self.vo = _vo
        self.DatasetHandler = DatasetHandler

    @staticmethod
    def drive_roi(h: int, w: int, keep: float = 0.5, side: float = 0.03,
                  region: str = "top") -> np.ndarray:
        """Their `drive_roi_mask`, taking the real frame size instead of the
        576x1024 their driver passes as a literal.

        `region` is not theirs. Their slice is `[0 : h*keep]`, the top half, so
        the road surface never produces a feature; the function's name suggests
        the drivable region was meant, which would be the complementary slice.
        "bottom" is that reading and "full" drops the vertical cut entirely, so
        the choice can be measured rather than argued about.
        """
        m = np.zeros((h, w), np.uint8)
        lo, hi = int(w * side), int(w * (1 - side))
        if region == "top":
            rows = slice(0, int(h * keep))
        elif region == "bottom":
            rows = slice(int(h * (1 - keep)), h)
        elif region == "full":
            rows = slice(0, h)
        else:
            raise ValueError(f"unknown roi region {region!r}")
        m[rows, lo:hi] = 255
        return m

    def _mask_for(self, rgb: np.ndarray) -> np.ndarray:
        """Their `det_obj` mask of movable objects, intersected with the road
        region, at the real frame size."""
        res = self.det_model(rgb, verbose=False)[0]
        movable = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
        h, w = rgb.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        boxes = res.boxes
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                if int(boxes.cls[i].item()) in movable:
                    x1, y1, x2, y2 = (int(v) for v in boxes.data[i, :4].cpu().numpy())
                    mask[y1:y2, x1:x2] = 0
        mask[self.drive_roi(h, w, region=self.roi) == 0] = 0
        return mask

    def run_clip(self, img_dir: str, K, seed: int,
                 intrinsics_mode: str = "predicted") -> dict:
        """Estimated camera locations, plus the diagnostics their script drops."""
        torch = self.torch
        files = sorted(f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png")))
        rgbs, depths, Ks, masks = [], [], [], []

        cond_K = None
        if intrinsics_mode == "calibrated_both":
            cond_K = torch.from_numpy(np.asarray(K, np.float32))

        t0 = time.time()
        for f in files:
            rgb = np.array(Image.open(os.path.join(img_dir, f)).convert("RGB"))
            masks.append(self._mask_for(rgb))
            rgbs.append(rgb)
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
            with torch.no_grad():
                pred = self.depth_model.infer(rgb_t, cond_K)
            depths.append(pred["depth"].squeeze().float().cpu().numpy())
            Ks.append(pred["intrinsics"].squeeze(0).float().cpu().numpy())
        t_depth = time.time() - t0

        pred_K = np.stack(Ks)
        if intrinsics_mode in ("calibrated_slam", "calibrated_both"):
            Ks = [np.asarray(K, float)] * len(files)

        # -- their pipeline from here on, unmodified ------------------------
        t0 = time.time()
        handler = self.DatasetHandler(rgbs, depths, Ks)
        kp_list, des_list = self.vo.extract_features_dataset(handler.images, masks)
        matches, matches2 = self.vo.match_features_dataset(des_list)
        matches = self.vo.filter_matches_dataset(matches, 0.7, matches2)

        ok_flags, n_pts = [], []
        original = self.vo.estimate_motion

        def watched(match, kp1, kp2, k, depth1=None):
            out = original(match, kp1, kp2, k, depth1)
            ok_flags.append(bool(out[4]))
            n_pts.append(len(out[2]))
            return out

        np.random.seed(seed)
        self.vo.estimate_motion = watched
        try:
            trajectory, poses = self.vo.estimate_trajectory(
                matches, kp_list, handler.k, depth_maps=handler.depth_maps,
                dataset_handler=handler)
        finally:
            self.vo.estimate_motion = original
        t_slam = time.time() - t0

        locs = np.stack([trajectory[0, :], trajectory[2, :]], axis=1).astype(np.float32)
        ok = np.asarray(ok_flags, bool)
        return {
            "locs": locs,                     # (N, 2) x, z -- their pkl layout
            "poses": poses,
            "ok": ok,
            "n_matched_points": np.asarray(n_pts, np.int32),
            "n_matches": np.asarray([len(m) for m in matches], np.int32),
            "predicted_K": pred_K,
            "slam_K": np.asarray(Ks[0], float),
            "intrinsics_mode": intrinsics_mode,
            "seed": seed,
            "fail_rate": float((~ok).mean()) if ok.size else float("nan"),
            "depth_median": float(np.median([np.median(d) for d in depths])),
            "sec_depth": t_depth,
            "sec_slam": t_slam,
        }


SCALARS = ("intrinsics_mode", "seed", "fail_rate", "depth_median",
           "sec_depth", "sec_slam")


def save_result(res: dict, path: str) -> None:
    np.savez_compressed(
        path,
        locs=res["locs"],
        ok=res["ok"],
        n_matched_points=res["n_matched_points"],
        n_matches=res["n_matches"],
        predicted_K=res["predicted_K"],
        slam_K=res["slam_K"],
        poses_R=np.stack([p[0] for p in res["poses"]]),
        poses_t=np.stack([np.asarray(p[1]).ravel() for p in res["poses"]]),
        meta=json.dumps({k: res[k] for k in SCALARS}),
    )
