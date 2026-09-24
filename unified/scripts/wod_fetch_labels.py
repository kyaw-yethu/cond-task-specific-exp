"""Pull the non-pixel components (0.25 GB total) and push one tar per
component, mirroring the nuScenes raw/ layout."""
import warnings; warnings.filterwarnings("ignore")
import gcsfs, tarfile, os, subprocess, time

BUCKET = "waymo_open_dataset_v_2_0_1"
VOL    = "volume://vessl-storage/yethu-drive/assets/waymo/raw"
WORK   = "/root/wod_stage"
COMPS  = ["vehicle_pose", "camera_calibration", "camera_box", "stats"]

fs = gcsfs.GCSFileSystem(token="google_default")
os.makedirs(WORK, exist_ok=True)

for comp in COMPS:
    local = f"{WORK}/{comp}.tar"
    n = tot = 0
    with tarfile.open(local, "w") as tar:
        for split in ["training", "validation", "testing"]:
            try:
                ents = [e for e in fs.ls(f"{BUCKET}/{split}/{comp}", detail=True)
                        if e["name"].endswith(".parquet") and e["size"] > 0]
            except Exception as e:
                print(f"  {comp}/{split} list failed: {e}", flush=True)
                continue
            for e in ents:
                blob = fs.cat_file(e["name"])
                ti = tarfile.TarInfo(f"{split}/{e['name'].split('/')[-1]}")
                ti.size = len(blob); ti.mtime = 0
                import io as _io
                tar.addfile(ti, _io.BytesIO(blob))
                n += 1; tot += len(blob)
    print(f"{comp}: {n} files, {tot/1e6:.1f} MB", flush=True)
    r = subprocess.run(f'vessl storage copy-file "{local}" "{VOL}/"',
                       shell=True, capture_output=True, text=True)
    print(f"  upload rc={r.returncode}", flush=True)
    os.remove(local)
print("LABELS DONE", flush=True)
