"""Stream WOD v2 camera_image from GCS, keep FRONT only, push each scene tar
to yethu-drive, delete locally. Peak local disk stays a few hundred MB.

FRONT jpgs are copied verbatim out of the parquet: no decode, no re-encode.
Resumable -- shards already on the volume are skipped.
"""
import warnings; warnings.filterwarnings("ignore")
import gcsfs, pyarrow.parquet as pq, tarfile, io, os, sys, json, time
import subprocess, threading, queue, traceback

BUCKET = "waymo_open_dataset_v_2_0_1"
SPLITS = ["training", "validation", "testing"]
VOL    = "volume://vessl-storage/yethu-drive/assets/waymo"
WORK   = "/root/wod_stage"
NWORK  = int(os.environ.get("WOD_WORKERS", "8"))
FRONT  = 1
LOG    = "/root/wod_fetch.log"

# RLock, not Lock: the progress branch calls log(), which takes the
# same lock, and a plain Lock deadlocks on that re-entry.
lock = threading.RLock()
def log(msg):
    with lock:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")

def existing_shards():
    # WOD_FRESH=1 ignores what is already there, so every shard is rewritten.
    # Leave it unset when resuming an interrupted run.
    if os.environ.get("WOD_FRESH") == "1":
        return set()
    rc, out = sh(f'vessl storage list-files "{VOL}/cam_front_shards" 2>/dev/null')
    return {l.split()[0] for l in out.splitlines()
            if l.strip().startswith("cam_front_") and ".tar" in l}

def do_segment(fs, split, path, done_set):
    ctx = path.split("/")[-1].replace(".parquet", "")
    name = f"cam_front_{ctx}.tar"
    if name in done_set:
        return "skip", 0, 0
    local = f"{WORK}/{name}"
    f = pq.ParquetFile(fs.open(path, "rb"))
    rows = []
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg, columns=["key.camera_name",
                                          "key.frame_timestamp_micros",
                                          "[CameraImageComponent].image"])
        cams = t.column("key.camera_name").to_pylist()
        ts   = t.column("key.frame_timestamp_micros").to_pylist()
        img  = t.column("[CameraImageComponent].image").to_pylist()
        rows += [(ts[i], img[i]) for i in range(len(cams)) if cams[i] == FRONT]
    rows.sort(key=lambda r: r[0])
    nbytes = 0
    with tarfile.open(local, "w") as tar:          # uncompressed, as nuScenes
        for tstamp, blob in rows:
            member = tarfile.TarInfo(f"{ctx}__CAM_FRONT__{tstamp}.jpg")
            member.size = len(blob)
            member.mtime = 0
            tar.addfile(member, io.BytesIO(blob))
            nbytes += len(blob)
    rc, out = sh(f'vessl storage copy-file "{local}" "{VOL}/cam_front_shards/"')
    if rc != 0:
        os.remove(local)
        raise RuntimeError(f"upload failed {ctx}: {out[-300:]}")
    os.remove(local)
    return split, len(rows), nbytes

def main():
    os.makedirs(WORK, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="google_default")
    done = existing_shards()
    log(f"already on volume: {len(done)} shards")

    work = []
    for split in SPLITS:
        ents = [e for e in fs.ls(f"{BUCKET}/{split}/camera_image", detail=True)
                if e["name"].endswith(".parquet") and e["size"] > 0]
        work += [(split, e["name"]) for e in ents]
        log(f"{split}: {len(ents)} segments")
    log(f"total {len(work)} segments, {NWORK} workers")

    q = queue.Queue()
    for w in work:
        q.put(w)
    stats = {"done": 0, "skip": 0, "fail": 0, "frames": 0, "bytes": 0}
    t0 = time.time()

    def worker(wid):
        wfs = gcsfs.GCSFileSystem(token="google_default")
        while True:
            try:
                split, path = q.get_nowait()
            except queue.Empty:
                return
            try:
                kind, nf, nb = do_segment(wfs, split, path, done)
                with lock:
                    if kind == "skip":
                        stats["skip"] += 1
                    else:
                        stats["done"] += 1
                        stats["frames"] += nf
                        stats["bytes"] += nb
                    n = stats["done"] + stats["skip"]
                    if n % 10 == 0:
                        el = time.time() - t0
                        rate = stats["bytes"] / el / 1e6 if el else 0
                        eta = (len(work) - n) * el / max(n, 1) / 3600
                        log(f"{n}/{len(work)}  done={stats['done']} "
                            f"skip={stats['skip']} fail={stats['fail']}  "
                            f"{stats['bytes']/1e9:.1f} GB kept  "
                            f"{rate:.0f} MB/s  ETA {eta:.1f} h")
            except Exception as e:
                with lock:
                    stats["fail"] += 1
                log(f"FAIL {path.split('/')[-1]}: {type(e).__name__}: {e}")
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(NWORK)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    el = time.time() - t0
    log(f"FINISHED in {el/3600:.2f} h  done={stats['done']} skip={stats['skip']} "
        f"fail={stats['fail']}  frames={stats['frames']}  "
        f"kept={stats['bytes']/1e9:.1f} GB")

if __name__ == "__main__":
    main()
