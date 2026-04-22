"""
Inference script for MLP-only model (no hash encoding)
Based on infer_slices_ngp.py, adapted for MLPOnlyModel
"""
import argparse
import os
import numpy as np
import torch
import tifffile

# Import the MLP-only model from mlp_only.py
from mlp_only import MLPOnlyModel, PositionalEncoding


def read_volume_shape(tif_path: str):
    """Get volume shape without loading all data"""
    try:
        with tifffile.TiffFile(tif_path) as tf:
            shape = tf.series[0].shape
            if len(shape) != 3:
                raise ValueError(f"Expected 3D volume, got shape {shape}")
            return tuple(int(x) for x in shape)  # (z, y, x)
    except Exception:
        vol = tifffile.imread(tif_path)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return tuple(int(x) for x in vol.shape)


def build_plane_normalized_coords(z_index: int, height: int, width: int, full_dims, device: torch.device) -> torch.Tensor:
    """
    Build normalized coordinates for a single z-plane
    Returns coordinates in [0, 1] range with shape [height*width, 3] in (x, y, z) order
    """
    vz, vy, vx = full_dims
    
    ys = torch.arange(0, height, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, device=device, dtype=torch.float32)
    
    # Normalize to [0, 1] using global dims
    ys_n = ys / (vy - 1)
    xs_n = xs / (vx - 1)
    z_n_val = float(z_index) / (vz - 1)
    
    grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    grid_z = torch.full((height, width), z_n_val, device=device)
    
    # Stack in (x, y, z) order for the model
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    coords = coords.view(-1, 3)
    
    return coords


def predict_plane(coords_flat: torch.Tensor, model: MLPOnlyModel, batch_size: int = 500000,
                  pixel_samples: int = 1, pixel_scale: torch.Tensor = None):
    """
    Predict density values for a plane of coordinates
    Uses batching to avoid OOM for large planes
    """
    num_coords = coords_flat.shape[0]
    device = coords_flat.device
    
    if pixel_samples > 1 and pixel_scale is not None:
        # Anti-aliased inference
        n_dims = coords_flat.shape[1]
        pixel_jitter = (torch.rand(num_coords, pixel_samples, n_dims, device=device) - 0.5) * pixel_scale.to(device)
        coords_expanded = coords_flat.unsqueeze(1) + pixel_jitter
        coords_expanded = torch.clamp(coords_expanded, 0.0, 1.0)
        coords_flat_expanded = coords_expanded.view(-1, n_dims)
        
        predictions = []
        for i in range(0, coords_flat_expanded.shape[0], batch_size):
            batch_coords = coords_flat_expanded[i:i+batch_size]
            batch_pred, _ = model(batch_coords, variance=None)
            predictions.append(batch_pred)
        
        pred_cat = torch.cat(predictions, dim=0)
        pred_cat = pred_cat.view(num_coords, pixel_samples).mean(dim=1)
        return pred_cat, None
    else:
        # Single sample per pixel
        predictions = []
        
        for i in range(0, num_coords, batch_size):
            batch_coords = coords_flat[i:i+batch_size]
            batch_pred, _ = model(batch_coords, variance=None)
            predictions.append(batch_pred)
        
        pred_cat = torch.cat(predictions, dim=0)
        return pred_cat, None


def choose_slices(dz: int, num_slices: int, seed: int, mode: str = "random"):
    """Choose z-slices to infer"""
    return [700]


def main():
    parser = argparse.ArgumentParser(description="Infer z-slices from a trained MLP-only model")
    parser.add_argument("--volume_tif", type=str, default="/workspace/LSM/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/mlp_only.pth")
    parser.add_argument("--out_dir", type=str, default="/workspace/utils/3d/data/ablation")
    parser.add_argument("--num_slices", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=500000)
    parser.add_argument("--slice_mode", type=str, default="fixed", choices=["random", "fixed", "range"])
    parser.add_argument("--psf_path", type=str, default="/workspace/LSM/psf_t0_v0.tif", help="Path to PSF file for re-blurring")
    parser.add_argument("--z_pad", type=int, default=20, help="Number of slices to extract before/after target slice for 3D convolution")
    parser.add_argument("--pixel_samples", type=int, default=1, help="Number of samples per pixel for anti-aliased inference")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Get volume shape
    volume = tifffile.imread(args.volume_tif)
    dz, dy, dx = volume.shape
    print(f"Reference volume shape: (z, y, x) = {(dz, dy, dx)}")
    
    # Load PSF for re-blurring
    print(f"Loading PSF from: {args.psf_path}")
    psf = tifffile.imread(args.psf_path).astype(np.float32)
    psf /= psf.sum()
    print(f"PSF shape: {psf.shape}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get model config
    model_config = ckpt.get('model_config', None)
    
    if model_config is None:
        # Use defaults if not found
        model_config = {
            "hidden_dim": 64,
            "num_layers": 4,
            "n_frequencies": 5,
            "n_input_dims": 3,
            "learn_variance": False,
        }
        print("Warning: model_config not found in checkpoint, using defaults")
    else:
        print(f"Loaded model config: {model_config}")
    
    # Reconstruct MLP-only model
    model = MLPOnlyModel(
        n_input_dims=model_config.get("n_input_dims", 3),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 4),
        n_frequencies=model_config.get("n_frequencies", 5),
        learn_variance=model_config.get("learn_variance", False),
        skip_connections=[4] if model_config.get("num_layers", 4) > 4 else [],
    ).to(device)
    
    # Load model weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully")
    
    # Choose slices
    z_indices = choose_slices(dz, args.num_slices, args.seed, mode=args.slice_mode)

    with torch.no_grad():
        for z_idx in z_indices:
            # Define the range of slices to infer (target ± z_pad)
            z_start = max(0, z_idx - args.z_pad)
            z_end = min(dz, z_idx + args.z_pad + 1)
            num_slices_chunk = z_end - z_start
            
            print(f"\nProcessing target slice z={z_idx}")
            print(f"  Inferring slices {z_start} to {z_end-1} ({num_slices_chunk} slices)")
            
            # Infer the volume chunk
            volume_chunk = np.zeros((num_slices_chunk, dy, dx), dtype=np.float32)
            
            # Compute pixel scale for anti-aliased inference
            pixel_scale = torch.tensor([1.0 / (dx - 1), 1.0 / (dy - 1), 1.0 / (dz - 1)])
            
            if args.pixel_samples > 1:
                print(f"  Using {args.pixel_samples} samples per pixel for anti-aliased inference")
            
            for i, z in enumerate(range(z_start, z_end)):
                # Build coordinates
                coords_flat = build_plane_normalized_coords(z, dy, dx, (dz, dy, dx), device)
                
                # Predict
                pred_flat, _ = predict_plane(coords_flat, model, batch_size=args.batch_size,
                                             pixel_samples=args.pixel_samples, pixel_scale=pixel_scale)
                
                # Reshape to 2D
                pred_plane = pred_flat.view(dy, dx)
                volume_chunk[i] = pred_plane.detach().float().cpu().numpy()
            
            print(f"  Volume chunk range: [{volume_chunk.min():.4f}, {volume_chunk.max():.4f}]")
            volume_chunk = np.clip(volume_chunk, 0.0, 1.0)
            
            # Re-blur with PSF using scipy.signal.fftconvolve
            from scipy.signal import fftconvolve
            blurred_chunk = fftconvolve(volume_chunk, psf, mode='same')
            
            # Extract the target slice from the blurred chunk
            slice_idx_in_chunk = z_idx - z_start
            blurred_slice = blurred_chunk[slice_idx_in_chunk]
            original_slice = volume_chunk[slice_idx_in_chunk]
            
            print(f"  Blurred slice range: [{blurred_slice.min():.4f}, {blurred_slice.max():.4f}]")
            
            # Scale to uint16 range
            blurred_slice = blurred_slice * 65535
            blurred_slice = np.clip(blurred_slice, 0.0, 65535.0)

            original_slice = original_slice * 65535
            original_slice = np.clip(original_slice, 0.0, 65535.0)

            # Convert and save
            out_img = blurred_slice.astype(np.float32)
            original_img = original_slice.astype(np.float32)

            # Save images
            checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
            out_path = os.path.join(args.out_dir, f"{checkpoint_name}_conv_z{z_idx:05d}.tif")
            tifffile.imwrite(out_path, out_img)
            
            #original_path = os.path.join(args.out_dir, f"{checkpoint_name}_z{z_idx:05d}.tif")
            #tifffile.imwrite(original_path, original_img)
            
            print(f"  Saved {out_path}")
#            print(f"  Saved {original_path}")
            print(f"    Output range: [{out_img.min()}, {out_img.max()}]")
            print(f"    Output shape: {out_img.shape}")


if __name__ == "__main__":
    main()
