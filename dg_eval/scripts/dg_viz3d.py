"""Export the 3D geometry the extractor actually builds, for the explainer page.

The four per-frame stages exist to produce one thing: a set of 3D points in the
camera's frame, each with a 2D observation in the next frame. PnP then solves for
the rigid motion that maps one onto the other, and composing 100 of those gives
the trajectory. None of that is visible in an image panel, so the points, the
solved motion and the composed pose chain go out as numbers for the page to draw.

Camera convention is OpenCV: x right, y down, z forward. The export negates y so
the page can treat it as up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from PIL import Image

from dg_eval.lib.extract import Extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/root/driving-gen/clips/dg/scene-0330")
    ap.add_argument("--traj", default="/root/driving-gen/traj/dg/predicted/scene-0330.npz")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--unidepth", default="/root/driving-gen/ckpt/unidepth-v2-vitl14")
    ap.add_argument("--yolo", default="/root/driving-gen/ckpt/yolov10x.pt")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--roi", default="top", choices=["top", "bottom", "full"])
    ap.add_argument("--max-points", type=int, default=700)
    ap.add_argument("--out", default="/root/driving-gen/viz/scene3d.json")
    args = ap.parse_args()

    with open(os.path.join(args.clip, "meta.json")) as f:
        meta = json.load(f)
    img_dir = os.path.join(args.clip, "images")
    ex = Extractor(args.repo, args.unidepth, args.yolo, roi=args.roi)
    import torch

    k = args.frame
    rgbs = [np.array(Image.open(os.path.join(img_dir, "%05d.jpg" % i)).convert("RGB"))
            for i in (k, k + 1)]
    h, w = rgbs[0].shape[:2]

    masks = [ex._mask_for(r) for r in rgbs]
    with torch.no_grad():
        pred = ex.depth_model.infer(torch.from_numpy(rgbs[0]).permute(2, 0, 1), None)
    depth = pred["depth"].squeeze().float().cpu().numpy()
    K = pred["intrinsics"].squeeze(0).float().cpu().numpy()
    Kinv = np.linalg.inv(K)

    grays = [cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) for r in rgbs]
    kp1, des1 = ex.vo.extract_features(grays[0], masks[0])
    kp2, des2 = ex.vo.extract_features(grays[1], masks[1])
    m, m2 = ex.vo.match_features(des1, des2)
    good = ex.vo.filter_matches_distance(m, 0.7, m2)

    pts, obs, keep, rgbcol = [], [], [], []
    for mm in good:
        u1, v1 = kp1[mm.queryIdx].pt
        u2, v2 = kp2[mm.trainIdx].pt
        s = float(depth[int(v1), int(u1)])
        ok = 1e-3 < s < 80
        p = Kinv @ (s * np.array([u1, v1, 1.0]))
        pts.append(p)
        obs.append([u1, v1, u2, v2])
        keep.append(bool(ok))
        rgbcol.append([int(c) for c in rgbs[0][int(v1), int(u1)]])
    pts = np.asarray(pts)
    keep = np.asarray(keep)
    print("matches %d | clearing depth threshold %d" % (len(good), keep.sum()))

    # the solve their code runs on exactly these points
    objp = pts[keep]
    imgp = np.asarray([o[2:] for o, kk in zip(obs, keep) if kk])
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objp, imgp, K, None)
    R, _ = cv2.Rodrigues(rvec)
    inl = np.zeros(len(objp), bool)
    if inliers is not None:
        inl[inliers[:, 0]] = True
    print("PnP: %d points, %d inliers, |t| = %.3f m" % (len(objp), inl.sum(),
                                                        float(np.linalg.norm(tvec))))

    # subsample for the page, keeping the pass/reject ratio honest
    rng = np.random.default_rng(0)
    idx_keep = np.where(keep)[0]
    idx_drop = np.where(~keep)[0]
    n_k = min(len(idx_keep), int(args.max_points * .62))
    n_d = min(len(idx_drop), args.max_points - n_k)
    sel = np.concatenate([rng.choice(idx_keep, n_k, replace=False),
                          rng.choice(idx_drop, n_d, replace=False)])
    sel.sort()

    # inlier flag back on the full index
    inl_full = np.zeros(len(pts), bool)
    inl_full[idx_keep[:len(inl)]] = inl

    far = 95.0   # park rejected points at a readable distance instead of infinity
    P = []
    for i in sel:
        x, y, z = pts[i]
        if not keep[i]:
            n = np.linalg.norm([x, y, z]) or 1.0
            x, y, z = np.array([x, y, z]) / n * far
        P.append([round(float(x), 2), round(float(-y), 2), round(float(z), 2),
                  1 if keep[i] else 0, 1 if inl_full[i] else 0,
                  round(obs[i][0] / w, 4), round(obs[i][1] / h, 4)])

    d = np.load(args.traj, allow_pickle=True)
    Rs, ts = d["poses_R"], d["poses_t"]
    chain = [[round(float(t[0]), 3), round(float(-t[1]), 3), round(float(t[2]), 3)]
             for t in ts]
    fwd = [[round(float(r[0, 2]), 3), round(float(-r[1, 2]), 3), round(float(r[2, 2]), 3)]
           for r in Rs]

    out = {
        "scene": meta["scene"], "frame": k, "w": w, "h": h,
        "K": [round(float(v), 2) for v in K.ravel()],
        "points": P,
        "n_matches": int(len(good)),
        "n_pass": int(keep.sum()),
        "n_inliers": int(inl.sum()),
        "step_R": [round(float(v), 5) for v in R.ravel()],
        "step_t": [round(float(v), 4) for v in np.asarray(tvec).ravel()],
        "step_len": round(float(np.linalg.norm(tvec)), 3),
        "chain": chain, "chain_fwd": fwd,
        "far": far,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote %s (%d points, %d poses, %.0f KB)"
          % (args.out, len(P), len(chain), os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
