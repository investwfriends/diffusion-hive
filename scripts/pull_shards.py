#!/usr/bin/env python3
"""High-speed background daemon to sync 50MB shards from remote cluster to local machine."""

import os
import subprocess
import time

REMOTE_HOST = "537bc3a3-41ff-4b60-a0dd-f1ca2b0af5f2@136.61.33.107"
REMOTE_PORT = "42203"
REMOTE_DIR = "/tf/DiffusionHive/data/shards"
LOCAL_DIR = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/shards_remote"

def pull_shards():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"[sync] High-speed rsync monitoring {REMOTE_HOST}:{REMOTE_DIR} -> {LOCAL_DIR}...", flush=True)
    
    while True:
        try:
            # High speed rsync of all completed .pt shards (excluding .tmp)
            rsync_cmd = [
                "rsync", "-avz", "--include=*.pt", "--exclude=*.tmp", "--exclude=*",
                "-e", f"ssh -o StrictHostKeyChecking=no -p {REMOTE_PORT}",
                f"{REMOTE_HOST}:{REMOTE_DIR}/", f"{LOCAL_DIR}/"
            ]
            subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
            
            synced_files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.pt')]
            total_size_mb = sum(os.path.getsize(os.path.join(LOCAL_DIR, f)) for f in synced_files) / (1024 * 1024)
            print(f"[sync] Local shards: {len(synced_files)} files ({total_size_mb:.1f} MB)", flush=True)
        except Exception as e:
            print(f"[sync] Warning during rsync loop: {e}", flush=True)
            
        time.sleep(10)

if __name__ == "__main__":
    pull_shards()
