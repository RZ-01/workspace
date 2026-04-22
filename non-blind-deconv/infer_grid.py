import argparse
import torch
import numpy as np
import tifffile

from models.instantngp import InstantNGPTorchModel


def infer_grid(model, resolution, device, batch_size=1000000, slice_size=128):
    """
    Infer grid in slices to avoid memory overflow.
    Process the grid in Z-slices to keep memory usage manageable.
    """
    print(f"Inferring grid on {resolution}³ grid (processing in {slice_size}-slice batches)...")
    
    # Pre-allocate output grid on CPU
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Create coordinate arrays on CPU first
    x = np.linspace(0, 1, resolution, dtype=np.float32)
    y = np.linspace(0, 1, resolution, dtype=np.float32)
    z = np.linspace(0, 1, resolution, dtype=np.float32)
    
    total_slices = (resolution + slice_size - 1) // slice_size
    
    # Process in Z-slices
    for slice_idx in range(total_slices):
        z_start = slice_idx * slice_size
        z_end = min(z_start + slice_size, resolution)
        actual_slice_size = z_end - z_start
        
        print(f"  Processing Z-slice {slice_idx+1}/{total_slices} (z={z_start}:{z_end})...")
        
        # Create meshgrid for this slice on CPU
        grid_z, grid_y, grid_x = np.meshgrid(
            z[z_start:z_end], 
            y, 
            x, 
            indexing='ij'
        )
        
        # Stack and reshape coordinates (keep on CPU)
        coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
        
        total_coords = coords.shape[0]
        slice_sdf_values = []
        
        # Process this slice in batches
        for i in range(0, total_coords, batch_size):
            # Only transfer the batch to GPU
            batch_coords = torch.from_numpy(coords[i:i+batch_size]).to(device)
            with torch.no_grad():
                batch_sdf, _ = model(batch_coords, variance=None)
            slice_sdf_values.append(batch_sdf.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Batch {i // batch_size + 1}/{(total_coords + batch_size - 1) // batch_size}")
        
        # Concatenate and reshape slice results
        slice_sdf = torch.cat(slice_sdf_values, dim=0)
        slice_sdf = slice_sdf.reshape(actual_slice_size, resolution, resolution).numpy()
        
        # Copy to output grid
        grid[z_start:z_end, :, :] = slice_sdf
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print("inference complete!")
    return grid

def main():
    parser = argparse.ArgumentParser(description="Extract occupancy grid from trained network")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/dragon.pth", help="Path to trained model checkpoint")
    parser.add_argument("--resolution", type=int, default=2048, help="Grid resolution")
    parser.add_argument("--output", type=str, default="/workspace/ours_dragon.tif", help="Output tif file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1000000, help="Batch size for inference")
    parser.add_argument("--slice_size", type=int, default=256, help="Number of Z-slices to process at once")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    encoder_config = ckpt.get('encoder_config')
    decoder_config = ckpt.get('decoder_config')
    
    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=3,
        learn_variance=False,
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Infer grid
    grid = infer_grid(model, args.resolution, device, args.batch_size, args.slice_size)
    
    # Clip to [0, 1] range
    grid = np.clip(grid, 0.0, 1.0)
    
    print(f"\nGrid statistics:")
    print(f"  Shape: {grid.shape}")
    print(f"  Min: {grid.min():.6f}")
    print(f"  Max: {grid.max():.6f}")
    print(f"  Mean: {grid.mean():.6f}")
    
    # Save as tif
    tifffile.imwrite(args.output, grid.astype(np.float32))
    print(f"\nSaved grid to: {args.output}")


if __name__ == "__main__":
    main()
