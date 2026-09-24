"""Compressed copy of womd_7hz_f33 on yethu-drive.

Each split directory is tarred through zstd into 4 GB parts under
/opt/womd/pack and pushed to volume://vessl-storage/yethu-drive/womd/womd_7hz_f33/.
Restore: cat <split>.tar.zst.part-* | zstd -d | tar -x
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cond_eval.unified.volume import open_volume, push

OUT = "data/womd_7hz_f33"
PACK = "/opt/womd/pack"
DST = "womd/womd_7hz_f33"

os.makedirs(PACK, exist_ok=True)
vol = open_volume(write=True)
manifest = {}
for sp in ("train", "val", "test"):
    prefix = os.path.join(PACK, sp + ".tar.zst.part-")
    if not glob.glob(prefix + "*"):
        cmd = ("tar -C %s -cf - %s | zstd -T64 -6 -q | split -b 4G -d -a 2 - %s" % (OUT, sp, prefix))
        subprocess.run(cmd, shell=True, check=True)
    parts = sorted(glob.glob(prefix + "*"))
    manifest[sp] = []
    for p in parts:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        push(vol, p, DST + "/" + os.path.basename(p))
        manifest[sp].append(dict(file=os.path.basename(p), bytes=os.path.getsize(p), md5=h.hexdigest()))
        print("pushed", os.path.basename(p), os.path.getsize(p) / 1e9, "GB", flush=True)
for f in ("README.md", "checks.json"):
    if os.path.exists(os.path.join(OUT, f)):
        push(vol, os.path.join(OUT, f), DST + "/" + f)
json.dump(manifest, open(os.path.join(PACK, "manifest.json"), "w"), indent=2)
push(vol, os.path.join(PACK, "manifest.json"), DST + "/manifest.json")
print(json.dumps(manifest, indent=1))
