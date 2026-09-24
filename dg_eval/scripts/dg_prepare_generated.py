"""Turn generated videos into clips the evaluation suite can read.

The whole chain (`dg_extract`, `dg_score`, `dg_ftd`, `dg_full_table`,
`plot_dg_traj`) reads one layout: a directory of clip folders, each holding
`images/00000.jpg ...` and a `meta.json`. This writes that layout from generated
output, taking the ground truth from the real clip the generation was
conditioned on.

Input per scene, either form:

    <gen>/scene-0330.mp4          a video
    <gen>/scene-0330/*.png        a directory of frames

Two things are recomputed rather than copied. **Intrinsics**, because a
generated clip is usually a different size from the real one it came from, and
`K` has to be derived from the native $1600\\times900$ calibration through the
crop and resize that reach the generated frame, not rescaled from an already
scaled matrix. And the **frame count**, because a generator usually produces
fewer frames than the source clip holds; the ground truth is truncated to match,
taking the frames from the start since image-conditioned generation begins at
frame 0.

Nothing here invents ground truth. A clip whose scene is absent from the real
set is skipped and named in the summary, because without a true ego path only
the reference-free metrics and FTD would mean anything.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image

from dg_eval.lib import clips as C
from dg_eval.lib import nusc_index as NI

VIDEO_EXT = (".mp4", ".avi", ".mov", ".webm", ".mkv")
FRAME_EXT = (".png", ".jpg", ".jpeg")


def discover(gen_dir):
    """Scene name -> the video file or frame directory holding its frames."""
    out = {}
    for entry in sorted(os.listdir(gen_dir)):
        path = os.path.join(gen_dir, entry)
        stem, ext = os.path.splitext(entry)
        if os.path.isdir(path):
            if any(f.lower().endswith(FRAME_EXT) for f in os.listdir(path)):
                out[stem] = path
        elif ext.lower() in VIDEO_EXT:
            out[stem] = path
    return out


def extract_frames(src, dst_dir, quality=95):
    """Frames out of a video or a frame directory, renumbered from zero."""
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(dst_dir):
        os.remove(os.path.join(dst_dir, f))

    if os.path.isdir(src):
        names = sorted(f for f in os.listdir(src) if f.lower().endswith(FRAME_EXT))
        for i, name in enumerate(names):
            out = os.path.join(dst_dir, "%05d.jpg" % i)
            if name.lower().endswith((".jpg", ".jpeg")):
                shutil.copyfile(os.path.join(src, name), out)
            else:
                Image.open(os.path.join(src, name)).convert("RGB").save(
                    out, quality=quality)
        return len(names)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-q:v", "2", "-start_number", "0", os.path.join(dst_dir, "%05d.jpg")],
        check=True)
    return len([f for f in os.listdir(dst_dir) if f.endswith(".jpg")])


def frame_size(dst_dir):
    first = sorted(f for f in os.listdir(dst_dir) if f.endswith(".jpg"))[0]
    with Image.open(os.path.join(dst_dir, first)) as im:
        return im.width, im.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True,
                    help="directory of generated videos or frame directories")
    ap.add_argument("--out", required=True, help="clip directory to write")
    ap.add_argument("--real-clips", default="/root/driving-gen/clips/dg",
                    help="the real clips the generation was conditioned on")
    ap.add_argument("--index", default="/root/driving-gen/nuscenes/index.json",
                    help="nuScenes index, for the native camera calibration")
    ap.add_argument("--frames", type=int, default=0,
                    help="truncate to this many frames; 0 keeps what the "
                         "generator produced")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    found = discover(args.gen)
    if args.limit:
        found = dict(list(found.items())[:args.limit])
    if not found:
        print("no videos or frame directories under", args.gen)
        return
    print("found %d generated clips" % len(found), flush=True)

    index = NI.load_index(args.index)
    os.makedirs(args.out, exist_ok=True)

    written, skipped, sizes = 0, [], set()
    for scene, src in found.items():
        real_meta = os.path.join(args.real_clips, scene, "meta.json")
        if scene not in index or not os.path.exists(real_meta):
            skipped.append((scene, "no real clip to take ground truth from"))
            continue
        with open(real_meta) as f:
            real = json.load(f)

        clip_dir = os.path.join(args.out, scene)
        img_dir = os.path.join(clip_dir, "images")
        try:
            n_frames = extract_frames(src, img_dir)
        except Exception as e:
            skipped.append((scene, "%s: %s" % (type(e).__name__, e)))
            continue
        if n_frames < 12:
            skipped.append((scene, "only %d frames" % n_frames))
            continue

        n = min(n_frames, len(real["ego_xy"]))
        if args.frames:
            n = min(n, args.frames)
        if n < n_frames:
            for f in sorted(os.listdir(img_dir))[n:]:
                os.remove(os.path.join(img_dir, f))
        if n < len(real["ego_xy"]):
            pass  # ground truth is truncated below to match

        w, h = frame_size(img_dir)
        sizes.add((w, h))
        crop, sx, sy = C._resize_plan(1600, 900, w, h)
        K = C.scale_intrinsics(index[scene].K, crop, sx, sy)

        meta = {
            "scene": scene,
            "geometry": "generated",
            "source": os.path.relpath(src, args.gen),
            "width": w, "height": h, "n_frames": n,
            "frame_idx": real["frame_idx"][:n],
            "t_us": real["t_us"][:n],
            "grid_residual_ms": real["grid_residual_ms"][:n],
            "K": K.tolist(),
            "crop": list(crop),
            "ego_xy": real["ego_xy"][:n],
            "cam_xy": real["cam_xy"][:n],
            "ego_yaw": real["ego_yaw"][:n],
        }
        with open(os.path.join(clip_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        written += 1

    print("\nwrote %d clips to %s" % (written, args.out))
    if sizes:
        print("frame sizes seen: %s" % ", ".join("%dx%d" % s for s in sorted(sizes)))
        if len(sizes) > 1:
            print("  WARNING: mixed sizes. FVD and FTD are not comparable across them.")
    if written:
        any_meta = json.load(open(os.path.join(
            args.out, sorted(os.listdir(args.out))[0], "meta.json")))
        nf = any_meta["n_frames"]
        print("frames per clip: %d  ->  %d FTD windows of 11 at stride 10"
              % (nf, max(0, (nf - 11) // 10 + 1)))
        print("the floor in DG_TRAJECTORY_FLOOR.md was measured at 101 frames and "
              "1024x576;\nre-measure it at this geometry before comparing conditions.")
    for scene, why in skipped[:15]:
        print("  skip %s: %s" % (scene, why))
    if len(skipped) > 15:
        print("  ... and %d more" % (len(skipped) - 15))

    with open(os.path.join(args.out, "_prepare.json"), "w") as f:
        json.dump({"written": written, "skipped": skipped,
                   "sizes": sorted(sizes)}, f, indent=1)


if __name__ == "__main__":
    main()
