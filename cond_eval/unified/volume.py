"""Scoped access to the VESSL volume.

`vessl storage copy-file` ignores the path below the volume root when it
downloads, so a request for one file pulls the whole volume. The python API
underneath it does not, and exposes an S3 client besides, so every transfer
here goes through that.
"""
from __future__ import annotations

import os
import random
import time

from vessl.storage.file import VolumeFile
from vessl.storage.volume_v2 import _get_volume_with_federate

STORAGE = "vessl-storage"
VOLUME = "yethu-drive"


def open_volume(write: bool = False, retries: int = 6):
    """A federated handle on the volume.

    Federation is a POST to the VESSL API and a pool of workers asking at once
    gets refused, so the call is staggered and retried rather than taken as
    fatal.
    """
    last = None
    for a in range(retries):
        try:
            return _get_volume_with_federate(
                STORAGE, VOLUME, federation_type="write" if write else "read")
        except Exception as e:
            last = e
            time.sleep(1.0 + 2.0 * a + random.random() * 2.0)
    raise RuntimeError("federate %s: %s" % ("write" if write else "read", last))


def listing(vol, prefix: str) -> dict:
    """`{basename: size}` under a prefix. `vol.list` returns paths relative to
    the prefix it was given, which is what the download call wants back."""
    return {f.path: (f.size or 0) for f in vol.list(prefix)}


def fetch(vol, prefix: str, name: str, size: int, dst: str, retries: int = 4) -> int:
    """One file down, skipping a byte-identical local copy."""
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last = None
    for a in range(retries):
        try:
            vol.download_file(VolumeFile(path=prefix + "/" + name, size=size), dst)
            if os.path.getsize(dst) == size:
                return size
            last = RuntimeError("short read %d != %d" % (os.path.getsize(dst), size))
        except Exception as e:                       # transient S3 failures
            last = e
        time.sleep(1.5 * (a + 1))
    raise RuntimeError("download %s: %s" % (name, last))


def push(vol, local: str, dest: str, retries: int = 4) -> int:
    """One file up. `dest` is a path inside the volume, no `volume://` prefix."""
    last = None
    for a in range(retries):
        try:
            vol.upload_file(local, dest)
            return os.path.getsize(local)
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError("upload %s: %s" % (dest, last))
