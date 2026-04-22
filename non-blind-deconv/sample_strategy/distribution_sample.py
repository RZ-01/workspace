import numpy as np
import torch
import torch.nn.functional as F

BLOCK_SIZE = 16

def sample_importance_coords(
    importance_map: torch.Tensor,
    num_samples: int,
    volume_shape: tuple,
    tile_size: int = 16,
    num_tiles: int = 35,
):
    """
    Sample voxel coordinates based on tile-level importance map.
    
    Strategy:
    1. Sample num_tiles tiles based on importance weights
    2. Distribute num_samples voxels across these tiles
    3. Uniformly sample voxels within each selected tile
    4. Clamp coordinates to valid volume bounds
    
    Args:
        importance_map: Tile-level importance map of shape (nZ, nY, nX)
        num_samples: Number of voxel coordinates to sample
        volume_shape: Shape of the actual volume (Z, Y, X) for bounds checking
        tile_size: Size of each tile (default: 16)
        num_tiles: Number of tiles to sample (default: 35)
    
    Returns:
        Voxel coordinates of shape (num_samples, 3) in (z, y, x) order
    """
    device = importance_map.device
    Z, Y, X = importance_map.shape
    vol_z, vol_y, vol_x = volume_shape
    
    # Step 1: Sample tiles based on importance
    probs = importance_map.reshape(-1)
    probs = probs / probs.sum()
    tile_idx = torch.multinomial(probs, num_tiles, replacement=True)
    
    # Convert flat tile indices to 3D tile coordinates
    zy = tile_idx // X
    x_tile = tile_idx % X
    z_tile = zy // Y
    y_tile = zy % Y
    
    # Step 2: Distribute voxels across tiles (approximately equal)
    voxels_per_tile = num_samples // num_tiles
    remaining = num_samples % num_tiles
    
    # Create list to store all voxel coordinates
    all_coords = []
    
    for i in range(num_tiles):
        # Number of voxels to sample from this tile
        n_voxels = voxels_per_tile + (1 if i < remaining else 0)
        
        # Calculate valid range for this tile (handle edge tiles)
        z_start = z_tile[i] * tile_size
        y_start = y_tile[i] * tile_size
        x_start = x_tile[i] * tile_size
        
        z_max = min(tile_size, vol_z - z_start)
        y_max = min(tile_size, vol_y - y_start)
        x_max = min(tile_size, vol_x - x_start)
        
        # Sample voxels uniformly within valid range of this tile
        z_offset = torch.randint(0, z_max, (n_voxels,), device=device)
        y_offset = torch.randint(0, y_max, (n_voxels,), device=device)
        x_offset = torch.randint(0, x_max, (n_voxels,), device=device)
        
        # Convert to absolute voxel coordinates
        z_voxel = z_start + z_offset
        y_voxel = y_start + y_offset
        x_voxel = x_start + x_offset
        
        coords = torch.stack([z_voxel, y_voxel, x_voxel], dim=1)
        all_coords.append(coords)
    
    # Concatenate all coordinates
    final_coords = torch.cat(all_coords, dim=0)
    
    return final_coords


def precompute_distribution_blocked(volume, save_path=None):
    print(f"Original volume shape: {volume.shape}")
    
    d, h, w = volume.shape
    pad_d = (BLOCK_SIZE - d % BLOCK_SIZE) % BLOCK_SIZE
    pad_h = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    
    volume_padded = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    d_p, h_p, w_p = volume_padded.shape
    
    sh = (d_p // BLOCK_SIZE, BLOCK_SIZE, 
          h_p // BLOCK_SIZE, BLOCK_SIZE, 
          w_p // BLOCK_SIZE, BLOCK_SIZE)
    
    view = volume_padded.reshape(sh)
    
    block_mass = view.sum(axis=(1, 3, 5)).astype(np.float32)
    
    print(f"Compressed shape (Block Grid): {block_mass.shape}")
    
    # Compute block-level probabilities
    prob = block_mass.flatten()
    total_mass = prob.sum()
    if total_mass > 0:
        prob = prob / total_mass
    else:
        prob = np.ones_like(prob) / prob.size
    
    
    data_to_save = {
        'prob': prob,
        'original_shape': np.array([d, h, w]),
        'grid_shape': np.array(block_mass.shape),
        'block_size': BLOCK_SIZE
    }
    
    if save_path:
        np.save(save_path, data_to_save)
        print(f"Saved blocked distribution to {save_path}")
    
    return data_to_save

def sample_distribution_coords_blocked(num_pixels, meta_data, device='cpu'):
    """
    Sample coordinates based on block-level distribution with uniform sampling within blocks.
    
    Args:
        num_pixels: Number of coordinates to sample
        meta_data: Dictionary containing block-level probabilities and metadata
        device: Target device for the output tensor
    
    Returns:
        torch.Tensor: Sampled coordinates of shape (num_pixels, 3)
    """
    prob = meta_data['prob'].astype(np.float32)
    grid_shape = meta_data['grid_shape']
    original_shape = meta_data['original_shape']
    block_size = meta_data['block_size']
    
    # Step 1: Sample blocks based on block-level probabilities
    cdf = np.cumsum(prob)
    
    rand_vals = np.random.random(num_pixels)
    block_indices_flat = np.searchsorted(cdf, rand_vals)
    block_indices_flat = np.clip(block_indices_flat, 0, len(cdf) - 1)
    
    z_blocks, y_blocks, x_blocks = np.unravel_index(block_indices_flat, grid_shape)
    
    z_start = z_blocks * block_size
    y_start = y_blocks * block_size
    x_start = x_blocks * block_size
    
    # Calculate the actual valid size for each block (handles edge blocks)
    z_max_size = np.minimum(block_size, original_shape[0] - z_start)
    y_max_size = np.minimum(block_size, original_shape[1] - y_start)
    x_max_size = np.minimum(block_size, original_shape[2] - x_start)
    
    # Step 2: Uniform sampling within each block
    z_offset = np.array([np.random.randint(0, max_sz) for max_sz in z_max_size])
    y_offset = np.array([np.random.randint(0, max_sz) for max_sz in y_max_size])
    x_offset = np.array([np.random.randint(0, max_sz) for max_sz in x_max_size])
    
    z_final = z_start + z_offset
    y_final = y_start + y_offset
    x_final = x_start + x_offset
    
    coords = np.stack([z_final, y_final, x_final], axis=1)
    
    return torch.from_numpy(coords).long().to(device)


if __name__ == "__main__":
    import argparse
    import tifffile
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input Tiff Volume")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path")
    args = parser.parse_args()
    
    print(f"Loading volume from {args.input}...")
    volume = tifffile.imread(args.input)
    
    meta_data = precompute_distribution_blocked(volume, args.output)
    
