"""Section 8 visual spot check: 20 clips per split, frames 0, 8, 16, 24, 32, one row per clip."""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

OUT = "data/womd_7hz_f33"
SLICES = ("turn", "lead_closing", "braking", "vru_nearby", "cruising", "static")
os.makedirs(os.path.join(OUT, "spotcheck"), exist_ok=True)
for sp in ("train", "val", "test"):
    v = np.load(os.path.join(OUT, sp, "videos.npy"), mmap_mode="r")
    a = np.load(os.path.join(OUT, sp, "annotations.npz"))
    idx = np.linspace(0, len(v) - 1, 20).round().astype(int)
    cols = (0, 8, 16, 24, 32)
    sheet = Image.new("RGB", (96 * len(cols) + 150, 96 * len(idx)), "white")
    d = ImageDraw.Draw(sheet)
    for r, i in enumerate(idx):
        for c, t in enumerate(cols):
            sheet.paste(Image.fromarray(np.ascontiguousarray(v[i][:, t].transpose(1, 2, 0))), (96 * c, 96 * r))
        txt = "#%d %s\nv=%.1f m/s\nintent=%d tl=%d" % (i, SLICES[int(a["slice"][i])], float(a["ego_speed"][i].mean()),
                                                  int(a["route_intent"][i]), int(a["tl_state"][i][0]))
        d.text((96 * len(cols) + 6, 96 * r + 10), txt, fill="black")
    sheet.save(os.path.join(OUT, "spotcheck", sp + ".png"))
    print("wrote", sp)
