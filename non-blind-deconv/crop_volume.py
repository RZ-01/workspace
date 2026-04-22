#!/usr/bin/env python3
"""
Crop a volume to a smaller region for local experiments.
Saves both .pt (for training) and .tif (for visualization).
"""

import argparse
import torch
import tifffile
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Crop volume to a smaller region")
    parser.add_argument("--input", type=str, default="/workspace/lsm/3.tif", help="Input volume path (.pt or .tif)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    # Load volume metadata using memory mapping
    print(f"Loading volume from {args.input}...")
    vol = tifffile.memmap(args.input, mode='r')
    print(f"Original volume shape: {vol.shape}")
    print(f"Original volume dtype: {vol.dtype}")
    
    # Determine output path first
    if args.output is None:
        base = args.input.rsplit('.', 1)[0]
        output_base = f"{base}"
    else:
        output_base = args.output
    
    # Create memory-mapped temporary file to avoid allocating 67 GiB upfront
    temp_file = f"{output_base}_temp.dat"
    vol_normalized = np.memmap(temp_file, dtype=np.float32, mode='w+', shape=vol.shape)
    
    # Process in chunks to reduce memory usage
    chunk_size = 50  # Process 50 Z-slices at a time
    
    print(f"Processing in chunks of {chunk_size} slices...")
    for i in range(0, vol.shape[0], chunk_size):
        end_idx = min(i + chunk_size, vol.shape[0])
        print(f"  Processing slices {i}-{end_idx-1}...")
        vol_normalized[i:end_idx] = vol[i:end_idx].astype(np.float32) / 65535.0
        vol_normalized.flush()  # Write to disk
    
    print(f"Original volume range: [{vol_normalized.min():.4f}, {vol_normalized.max():.4f}]")
    
    # Save as .pt (for training)
    pt_path = f"{output_base}.pt"
    print(f"Converting to tensor and saving to {pt_path}...")
    
    # Load in chunks to avoid memory issues when converting to tensor
    tensor_chunks = []
    for i in range(0, vol.shape[0], chunk_size):
        end_idx = min(i + chunk_size, vol.shape[0])
        tensor_chunks.append(torch.from_numpy(np.array(vol_normalized[i:end_idx])))
    
    vol_tensor = torch.cat(tensor_chunks, dim=0)
    torch.save(vol_tensor, pt_path)
    print(f"Saved .pt to: {pt_path}")
    
    # Clean up temporary file
    del vol_normalized
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Cleaned up temporary file: {temp_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
