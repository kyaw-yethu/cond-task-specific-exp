"""Fetch WOMD v1.3.0 training_20s shards for womd_7hz_f33.

Each shard goes GCS -> /opt/womd/raw (overlay disk, the renderer reads it from
there) -> volume://vessl-storage/yethu-drive/womd/raw/training_20s/.
Resumable: a shard already on the volume at the GCS size is skipped.
Shards: train pool 0-419, val 900-949, test 800-899 + 950-999.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, threading, queue
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gcsfs
from unified.lib.volume import open_volume, listing, push

SRC = "waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training_20s"
DST = "womd/raw/training_20s"
LOCAL = "/opt/womd/raw"
LOG = "data/womd_fetch.log"
SHARDS = list(range(0, 420)) + list(range(800, 1000))
NWORK = int(os.environ.get("NWORK", "8"))

lock = threading.Lock()
def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    with lock:
        print(line, flush=True)
        open(LOG, "a").write(line + "\n")

fs = gcsfs.GCSFileSystem(token="/root/.config/gcloud/application_default_credentials.json")
sizes = {os.path.basename(x["name"]): x["size"] for x in fs.ls(SRC, detail=True)}
names = ["training_20s.tfrecord-%05d-of-01000" % i for i in SHARDS]
vol0 = open_volume(write=False)
try:
    have = listing(vol0, DST)
except Exception:
    have = {}
todo = [n for n in names if have.get(n) != sizes[n]]
total = sum(sizes[n] for n in todo)
log("start: %d shards, %d already on volume, %d to fetch, %.1f GB" % (len(names), len(names) - len(todo), len(todo), total / 1e9))

q = queue.Queue()
for n in todo: q.put(n)
state = dict(done=0, bytes=0, fail=0); t0 = time.time()

def worker(k):
    time.sleep(k * 1.5)
    vol = open_volume(write=True)
    wfs = gcsfs.GCSFileSystem(token="/root/.config/gcloud/application_default_credentials.json")
    while True:
        try: n = q.get_nowait()
        except queue.Empty: return
        dst = os.path.join(LOCAL, n)
        try:
            if not (os.path.exists(dst) and os.path.getsize(dst) == sizes[n]):
                wfs.get(SRC + "/" + n, dst + ".part"); os.replace(dst + ".part", dst)
            assert os.path.getsize(dst) == sizes[n], "size mismatch"
            push(vol, dst, DST + "/" + n)
            with lock:
                state["done"] += 1; state["bytes"] += sizes[n]
                d, b = state["done"], state["bytes"]
            el = time.time() - t0; rate = b / el
            if d % 10 == 0 or d == len(todo):
                log("%d/%d  %.1f GB  %.0f MB/s  ETA %.2f h" % (d, len(todo), b / 1e9, rate / 1e6, (total - b) / rate / 3600))
        except Exception as e:
            with lock: state["fail"] += 1
            log("FAIL %s: %r" % (n, e))

ts = [threading.Thread(target=worker, args=(k,)) for k in range(NWORK)]
[t.start() for t in ts]; [t.join() for t in ts]
log("FINISHED in %.2f h  done=%d fail=%d  %.1f GB" % ((time.time() - t0) / 3600, state["done"], state["fail"], state["bytes"] / 1e9))
