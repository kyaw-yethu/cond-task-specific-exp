"""Render one clip through every stage of the extractor, for the explainer page.

Emits four JPEGs and one JSON of trajectories. The images are the stages that
are genuinely visual: the frame as it enters, the mask that decides which pixels
may produce features, the metric depth that supplies scale, and the surviving
correspondences that reach the PnP solve. The trajectories go out as numbers so
the page can draw them theme-aware.
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
from dg_eval.lib import score as S


def colorize_depth(d: np.ndarray, vmax: float = 60.0) -> np.ndarray:
    x = np.clip(d, 0, vmax) / vmax
    x = (x * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(255 - x, cv2.COLORMAP_TURBO)
    return bgr[:, :, ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/root/driving-gen/clips/dg/scene-0330")
    ap.add_argument("--traj", default="/root/driving-gen/traj/dg/predicted/scene-0330.npz")
    ap.add_argument("--repo", default="third_party/DrivingGen")
    ap.add_argument("--unidepth", default="/root/driving-gen/ckpt/unidepth-v2-vitl14")
    ap.add_argument("--yolo", default="/root/driving-gen/ckpt/yolov10x.pt")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--roi", default="top", choices=["top", "bottom", "full"])
    ap.add_argument("--out", default="/root/driving-gen/viz")
    ap.add_argument("--width", type=int, default=880)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.clip, "images")
    with open(os.path.join(args.clip, "meta.json")) as f:
        meta = json.load(f)

    ex = Extractor(args.repo, args.unidepth, args.yolo, roi=args.roi)
    import torch

    k = args.frame
    paths = [os.path.join(img_dir, "%05d.jpg" % i) for i in (k, k + 1)]
    rgbs = [np.array(Image.open(p).convert("RGB")) for p in paths]
    h, w = rgbs[0].shape[:2]

    def save(name, arr, quality=86):
        im = Image.fromarray(arr.astype(np.uint8))
        if im.width > args.width:
            im = im.resize((args.width, round(im.height * args.width / im.width)),
                           Image.LANCZOS)
        im.save(os.path.join(args.out, name), quality=quality)
        print("wrote", name, im.size)

    # -- 1. the frame as it enters -------------------------------------
    save("01_frame.jpg", rgbs[0])

    # -- 2. detections, road ROI, and the mask they produce ------------
    res = ex.det_model(rgbs[0], verbose=False)[0]
    movable = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    boxes = []
    if res.boxes is not None and len(res.boxes):
        for i in range(len(res.boxes)):
            if int(res.boxes.cls[i].item()) in movable:
                boxes.append([int(v) for v in res.boxes.data[i, :4].cpu().numpy()])

    mask = ex._mask_for(rgbs[0])
    over = rgbs[0].astype(np.float32)
    dead = (mask == 0)
    over[dead] = over[dead] * 0.22 + np.array([8, 12, 20]) * 0.78
    over = over.astype(np.uint8).copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(over, (x1, y1), (x2, y2), (255, 108, 40), 2)
    if args.roi != "full":
        roi_y = int(h * 0.5)
        cv2.line(over, (0, roi_y), (w, roi_y), (90, 220, 200), 2, cv2.LINE_AA)
    save("02_mask.jpg", over)

    # -- 3. metric depth ------------------------------------------------
    with torch.no_grad():
        pred = ex.depth_model.infer(torch.from_numpy(rgbs[0]).permute(2, 0, 1), None)
    depth = pred["depth"].squeeze().float().cpu().numpy()
    save("03_depth.jpg", colorize_depth(depth), quality=90)

    # -- 4. correspondences that survive to the PnP solve ---------------
    grays = [cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) for r in rgbs]
    mask2 = ex._mask_for(rgbs[1])
    kp1, des1 = ex.vo.extract_features(grays[0], mask)
    kp2, des2 = ex.vo.extract_features(grays[1], mask2)
    m, m2 = ex.vo.match_features(des1, des2)
    good = ex.vo.filter_matches_distance(m, 0.7, m2)

    canvas = (rgbs[0].astype(np.float32) * 0.42 + 26).astype(np.uint8).copy()
    n_pass = 0
    for mm in good:
        u1, v1 = kp1[mm.queryIdx].pt
        u2, v2 = kp2[mm.trainIdx].pt
        s = depth[int(v1), int(u1)]
        keep = 1e-3 < s < 80
        col = (70, 235, 205) if keep else (150, 150, 150)
        if keep:
            n_pass += 1
        cv2.line(canvas, (int(u1), int(v1)), (int(u2), int(v2)), col,
                 1 if keep else 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(u1), int(v1)), 2, col, -1, cv2.LINE_AA)
    save("04_matches.jpg", canvas, quality=90)
    print("matches after ratio test: %d | clearing depth gate: %d" % (len(good), n_pass))

    # -- 5. trajectories, as numbers ------------------------------------
    d = np.load(args.traj, allow_pickle=True)
    gt = np.asarray(meta["ego_xy"], float)
    fixed, ref, _ = S.prepare_pair(args.repo, d["locs"], gt, with_scale=False)
    scaled, _, s_opt = S.prepare_pair(args.repo, d["locs"], gt, with_scale=True)
    dgm = S.load_drivinggen(args.repo)
    out = {
        "scene": meta["scene"],
        "frame": k,
        "gt": np.round(ref, 3).tolist(),
        "fixed": np.round(fixed, 3).tolist(),
        "scaled": np.round(scaled, 3).tolist(),
        "scale": round(float(s_opt), 4),
        "ade_fixed": round(float(dgm.align.ade(fixed[None], ref[None]).ravel()[0]), 3),
        "ade_scaled": round(float(dgm.align.ade(scaled[None], ref[None]).ravel()[0]), 3),
        "n_matches": int(len(good)),
        "n_pnp": int(n_pass),
        "depth_median": round(float(np.median(depth)), 2),
        "n_boxes": len(boxes),
    }
    with open(os.path.join(args.out, "traj.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({k2: v for k2, v in out.items()
                      if k2 not in ("gt", "fixed", "scaled")}, indent=1))


if __name__ == "__main__":
    main()
