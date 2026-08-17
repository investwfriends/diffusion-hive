#!/usr/bin/env python3
import os
import random
import sys
import time
import torch

def main():
    path1 = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/dataset_merged.pt"
    path2 = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/ghive_diffusion_lite/dataset_merged.pt"
    output_path = "/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/dataset_combined.pt"

    start_time = time.time()

    print(f"Loading new dataset: {path1}")
    if not os.path.exists(path1):
        print(f"Error: {path1} not found.")
        sys.exit(1)
    
    t0 = time.time()
    data1 = torch.load(path1, map_location="cpu", weights_only=False)
    print(f"Loaded {len(data1)} samples in {time.time() - t0:.2f} seconds.")

    print(f"Loading old dataset: {path2}")
    if not os.path.exists(path2):
        print(f"Error: {path2} not found.")
        sys.exit(1)

    t0 = time.time()
    data2 = torch.load(path2, map_location="cpu", weights_only=False)
    print(f"Loaded {len(data2)} samples in {time.time() - t0:.2f} seconds.")

    print("Combining datasets...")
    combined = data1 + data2
    total_samples = len(combined)
    print(f"Total combined samples: {total_samples}")

    # Free up memory immediately
    del data1
    del data2

    print("Shuffling combined samples...")
    t0 = time.time()
    random.shuffle(combined)
    print(f"Shuffled in {time.time() - t0:.2f} seconds.")

    print(f"Saving combined dataset to: {output_path}")
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(combined, output_path)
    print(f"Saved successfully in {time.time() - t0:.2f} seconds.")

    final_size = os.path.getsize(output_path)
    print(f"Combined dataset file size: {final_size / 1024**3:.2f} GB ({final_size / 1024**2:.1f} MB)")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
