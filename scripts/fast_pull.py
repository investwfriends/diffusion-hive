#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

REMOTE_HOST = "537bc3a3-41ff-4b60-a0dd-f1ca2b0af5f2@136.61.33.107"
REMOTE_PORT = "42203"
REMOTE_DIR = "/tf/DiffusionHive/data/shards"
LOCAL_DIR = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/shards_remote"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-p", REMOTE_PORT,
        REMOTE_HOST, f"ls -1 {REMOTE_DIR}/*.pt"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    remote_files = [l.strip() for l in res.stdout.strip().split("\n") if l.strip().endswith(".pt")]
    existing = set(os.listdir(LOCAL_DIR))
    missing = [r for r in remote_files if os.path.basename(r) not in existing]
    
    print(f"Total Remote Shards: {len(remote_files)} | Already Local: {len(existing)} | Pulling: {len(missing)}")
    
    def pull(rpath):
        fn = os.path.basename(rpath)
        scp_cmd = [
            "scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
            f"{REMOTE_HOST}:{rpath}", os.path.join(LOCAL_DIR, fn)
        ]
        res = subprocess.run(scp_cmd, capture_output=True)
        if res.returncode == 0:
            print(f"  ✓ Downloaded {fn}")
        else:
            print(f"  ✗ Failed {fn}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(pull, missing))
        
    synced = [f for f in os.listdir(LOCAL_DIR) if f.endswith(".pt")]
    total_gb = sum(os.path.getsize(os.path.join(LOCAL_DIR, f)) for f in synced) / (1024**3)
    print(f"\n🎉 DONE! All {len(synced)} shards downloaded locally ({total_gb:.2f} GB total).")

if __name__ == "__main__":
    main()
