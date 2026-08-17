#!/usr/bin/env python3
import os
import random
import sys
import time
import torch

def merge_shards(shard_dir, output_file):
    print(f"Scanning directory: {shard_dir}")
    if not os.path.isdir(shard_dir):
        print(f"Error: {shard_dir} is not a directory.")
        sys.exit(1)

    # Find all shard_*.pt files (exclude dataset_merged or other things)
    files = [
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ]
    files.sort()

    total_files = len(files)
    if total_files == 0:
        print("No shard files found.")
        sys.exit(1)

    print(f"Found {total_files} shard files to merge.")

    all_samples = []
    start_time = time.time()

    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        if (idx + 1) % 50 == 0 or idx == 0 or idx == total_files - 1:
            print(f"[{idx + 1}/{total_files}] Loading {filename}...")
        try:
            samples = torch.load(filepath, map_location="cpu", weights_only=False)
            if isinstance(samples, list):
                all_samples.extend(samples)
            else:
                print(f"Warning: {filename} does not contain a list of samples. Skipped.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    load_time = time.time() - start_time
    total_samples = len(all_samples)
    print(f"Loaded {total_samples} samples in {load_time:.2f} seconds.")

    if total_samples == 0:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    print("Shuffling all samples...")
    shuffle_start = time.time()
    random.shuffle(all_samples)
    shuffle_time = time.time() - shuffle_start
    print(f"Shuffled in {shuffle_time:.2f} seconds.")

    print(f"Saving merged dataset to: {output_file}")
    save_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    torch.save(all_samples, output_file)
    save_time = time.time() - save_start
    print(f"Saved successfully in {save_time:.2f} seconds.")

    final_size = os.path.getsize(output_file)
    print(f"Merged dataset size: {final_size / 1024**3:.2f} GB ({final_size / 1024**2:.1f} MB)")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    default_dir = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/cloud_shards"
    default_out = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/dataset_merged.pt"
    
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_out
    
    merge_shards(shard_dir, output_file)
