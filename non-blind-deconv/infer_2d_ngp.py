import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

from models.instantngp import InstantNGPTorchModel

def variance_from_gt_psf(image_np, psf_np, device='cuda', tile_size=2048, overlap=64):
    """
    image_np: (H,W) float32 in [0,1] (or any scale, it's fine)
    psf_np:   (kh,kw) float32, MUST be normalized (sum==1)
    device:   'cuda' or 'cpu'
    tile_size: process image in tiles of this size to avoid OOM
    overlap: overlap between tiles to handle boundary effects
    returns:  (H,W) float32 variance map
    """
    H, W = image_np.shape
    
    # Move PSF to GPU
    K = torch.from_numpy(psf_np).float().to(device)
    K = K / (K.sum() + 1e-12)
    K4 = K[None, None]  # 1x1xkhxkw
    
    r_y, r_x = K.shape[0]//2, K.shape[1]//2
    pad = (r_x, r_x, r_y, r_y)
    
    # Initialize output arrays on CPU
    var_map = np.zeros((H, W), dtype=np.float32)
    
    print(f"Processing variance map in tiles of {tile_size}x{tile_size} with overlap {overlap}...")
    
    # Process image in tiles
    y_starts = list(range(0, H, tile_size - overlap))
    x_starts = list(range(0, W, tile_size - overlap))
    
    total_tiles = len(y_starts) * len(x_starts)
    tile_idx = 0
    
    for y_start in y_starts:
        y_end = min(y_start + tile_size, H)
        for x_start in x_starts:
            x_end = min(x_start + tile_size, W)
            
            tile_idx += 1
            if tile_idx % 10 == 0:
                print(f"  Processing tile {tile_idx}/{total_tiles}...")
            
            # Extract tile
            tile = image_np[y_start:y_end, x_start:x_end]
            
            # Move tile to GPU
            I = torch.from_numpy(tile).float().to(device)
            I4 = I[None, None]  # 1x1xHtxWt
            
            # Compute variance for this tile
            mu = F.conv2d(F.pad(I4, pad, mode='reflect'), K4).squeeze(0).squeeze(0)
            m2 = F.conv2d(F.pad((I*I)[None, None], pad, mode='reflect'), K4).squeeze(0).squeeze(0)
            var_tile = (m2 - mu**2).clamp_min(0.0)
            
            # Move result to CPU
            var_tile_np = var_tile.cpu().numpy()
            
            # Determine the region to copy (avoiding overlap)
            if y_start == 0:
                y_copy_start = 0
            else:
                y_copy_start = overlap // 2
            
            if x_start == 0:
                x_copy_start = 0
            else:
                x_copy_start = overlap // 2
            
            y_copy_end = var_tile_np.shape[0]
            x_copy_end = var_tile_np.shape[1]
            
            # Copy to output
            var_map[y_start + y_copy_start:y_end, 
                    x_start + x_copy_start:x_end] = var_tile_np[y_copy_start:y_copy_end, 
                                                                  x_copy_start:x_copy_end]
            
            # Free GPU memory
            del I, I4, mu, m2, var_tile
            torch.cuda.empty_cache()
    
    print("Variance map computation completed.")
    return var_map

def gaussian_psf_from_cov(Sigma, radius=3.0):
    """
    Sigma: 2x2 covariance (in PIXELS^2)
    radius: kernel extends to ~radius * max_std
    returns normalized (kh,kw) psf
    """
    Sigma = Sigma.cpu().numpy()
    evals, evecs = np.linalg.eigh(Sigma)
    std_max = float(np.sqrt(evals.max()))
    half = int(np.ceil(radius * std_max))
    ys = np.arange(-half, half+1, dtype=np.float32)
    xs = np.arange(-half, half+1, dtype=np.float32)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    P = np.stack([X, Y], axis=-1)                                     # HxWx2
    Sigma_inv = np.linalg.inv(Sigma)
    # exp(-0.5 * v^T Σ^-1 v)
    expo = -0.5 * np.einsum('...i,ij,...j->...', P, Sigma_inv, P)
    K = np.exp(expo).astype(np.float32)
    K /= K.sum() + 1e-12
    return K

def build_2d_normalized_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Build normalized coordinates for a 2D image
    Returns coordinates in [0, 1] range with shape [height*width, 2] in (x, y) order
    """
    ys = torch.arange(0, height, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, device=device, dtype=torch.float32)
    
    # Normalize to [0, 1]
    ys_n = ys / (height - 1) if height > 1 else torch.zeros_like(ys)
    xs_n = xs / (width - 1) if width > 1 else torch.zeros_like(xs)
    
    grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    
    # Stack in (x, y) order for the model
    coords = torch.stack([grid_x, grid_y], dim=-1)
    coords = coords.view(-1, 2)
    
    return coords


def predict_image(coords_flat: torch.Tensor, model: InstantNGPTorchModel, batch_size: int = 1000000):
    """
    Predict density values for 2D image coordinates
    Uses batching to avoid OOM for large images
    Returns: (predictions, variances) on CPU
    """
    num_coords = coords_flat.shape[0]
    predictions = []
    variances = []
    
    num_batches = (num_coords + batch_size - 1) // batch_size
    
    for i in range(0, num_coords, batch_size):
        batch_idx = i // batch_size + 1
        if batch_idx % 100 == 0 or batch_idx == num_batches:
            print(f"  Batch {batch_idx}/{num_batches}")
        
        batch_coords = coords_flat[i:i+batch_size]
        batch_pred, batch_var = model(batch_coords, variance=None)
        
        # Move to CPU immediately to free GPU memory
        predictions.append(batch_pred.cpu())
        if batch_var is not None:
            variances.append(batch_var.cpu())
        
        # Clear GPU cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate on CPU
    pred_cat = torch.cat(predictions, dim=0)
    var_cat = torch.cat(variances, dim=0) if len(variances) > 0 else None
    
    return pred_cat, var_cat


def calculate_variance_mc_2d(coords_flat: torch.Tensor, model: InstantNGPTorchModel, 
                              num_samples: int = 2000, batch_size: int = 1800000, 
                              L: torch.Tensor = None, img_dims: tuple = None,
                              coord_batch_size: int = 100000) -> torch.Tensor:
    """
    Calculate variance using Monte Carlo sampling for 2D images
    Uses batching over coordinates to avoid OOM
    """
    num_coords = coords_flat.shape[0]
    device = coords_flat.device
    
    # Initialize accumulators for E(x) and E(x^2) - store on CPU to save GPU memory
    sum_x = torch.zeros(num_coords, dtype=torch.float32)
    sum_x2 = torch.zeros(num_coords, dtype=torch.float32)
    
    print(f"Calculating variance with {num_samples} Monte Carlo samples...")
    print(f"Processing {num_coords} coordinates in batches of {coord_batch_size}...")
    
    height, width = img_dims
    norm_factor = torch.tensor([width - 1, height - 1], device=device, dtype=torch.float32)
    
    # Move L to device
    L_device = L.to(device)
    
    # Process coordinates in batches to avoid OOM
    num_coord_batches = (num_coords + coord_batch_size - 1) // coord_batch_size
    
    for coord_batch_idx in range(num_coord_batches):
        coord_start = coord_batch_idx * coord_batch_size
        coord_end = min(coord_start + coord_batch_size, num_coords)
        batch_num_coords = coord_end - coord_start
        
        coords_batch = coords_flat[coord_start:coord_end]
        
        # Initialize accumulators for this coordinate batch
        batch_sum_x = torch.zeros(batch_num_coords, device=device)
        batch_sum_x2 = torch.zeros(batch_num_coords, device=device)
        
        print(f"  Coordinate batch {coord_batch_idx + 1}/{num_coord_batches} (coords {coord_start}-{coord_end})")
        
        for sample_idx in range(num_samples):
            if (sample_idx + 1) % 500 == 0:
                print(f"    Sample {sample_idx + 1}/{num_samples}")

            # Sample independent PSF offsets for each coordinate point in pixel space
            standard_normal = torch.randn(batch_num_coords, 2, device=device)
            psf_offset_pixel = standard_normal @ L_device.T 
            
            # Normalize PSF offset from pixel space to [0, 1] space
            psf_offset_normalized = psf_offset_pixel / norm_factor
            
            # Add offset and clamp to valid range [0, 1]
            coords_sampled = (coords_batch + psf_offset_normalized).clamp(0.0, 1.0)
        
            # Predict in sub-batches if needed
            predictions = []
            for i in range(0, batch_num_coords, batch_size):
                sub_batch_coords = coords_sampled[i:i+batch_size]
                batch_pred, _ = model(sub_batch_coords, variance=None)
                predictions.append(batch_pred)
            
            pred = torch.cat(predictions, dim=0)
            
            batch_sum_x += pred
            batch_sum_x2 += pred ** 2
        
        # Move batch results to CPU and accumulate
        sum_x[coord_start:coord_end] = batch_sum_x.cpu()
        sum_x2[coord_start:coord_end] = batch_sum_x2.cpu()
    
    # Calculate statistics on CPU
    E_x = sum_x / num_samples
    E_x2 = sum_x2 / num_samples
    
    # Calculate variance: Var(x) = E(x^2) - E(x)^2
    variance = E_x2 - E_x ** 2
    variance = torch.clamp(variance, min=0.0)
    
    print(f"Variance statistics: min={variance.min().item():.6e}, max={variance.max().item():.6e}, mean={variance.mean().item():.6e}")
    
    # Return on device
    return variance.to(device)


def main():
    parser = argparse.ArgumentParser(description="Infer 2D image from a trained Instant-NGP network and save as PNG")
    parser.add_argument("--input_image", type=str, default="/workspace/city.png", 
                        help="Input image path (PNG, JPG, etc.)")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/city.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--out_dir", type=str, default="/workspace/utils/2d/data",
                        help="Output directory for results")
    parser.add_argument("--psf_covariance_path", type=str, default="/workspace/sigma_20_noisy/gaussian_cov.pt",
                        help="Path to PSF covariance matrix (for variance computation)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=100000,
                        help="Batch size for inference")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="Scale factor to multiply predictions")
    parser.add_argument("--compute_variance", action="store_true",
                        help="Compute and visualize variance map")
    parser.add_argument("--mc_samples", type=int, default=2000,
                        help="Number of Monte Carlo samples for variance")

    parser.add_argument("--coord_batch_size", type=int, default=100000,
                        help="Batch size for coordinates during variance computation (reduce if OOM)")
    parser.add_argument("--variance_tile_size", type=int, default=2048,
                        help="Tile size for variance computation from GT PSF (reduce if OOM)")
    parser.add_argument("--variance_overlap", type=int, default=64,
                        help="Overlap between tiles for variance computation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load input image
    print(f"Loading image: {args.input_image}")
    img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image from {args.input_image}")
    
    # Get image dimensions
    if len(img.shape) == 3:
        # Color image - convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Converted color image to grayscale")
    
    height, width = img.shape
    print(f"Image shape: (height, width) = ({height}, {width})")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get encoder and decoder configs
    encoder_config = ckpt.get('encoder_config', None)
    decoder_config = ckpt.get('decoder_config', None)
    
    if encoder_config is None or decoder_config is None:
        print("Warning: No encoder/decoder config found in checkpoint, using defaults")
        encoder_config = None
        decoder_config = None
    else:
        print(f"Loaded encoder config: {encoder_config}")
        print(f"Loaded decoder config: {decoder_config}")
    
    # Reconstruct model with 2D input
    model = InstantNGPTorchModel(
        encoder_config=encoder_config, 
        decoder_config=decoder_config,
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    
    # Load model weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    with torch.no_grad():
        # Build 2D coordinates
        print("Building normalized coordinates...")
        coords_flat = build_2d_normalized_coords(height, width, device)
        print(f"Coordinate shape: {coords_flat.shape}")
        
        # Predict
        print("Running inference...")
        pred_flat, model_var_flat = predict_image(coords_flat, model, batch_size=args.batch_size)
        
        # Reshape to 2D image
        pred_img = pred_flat.view(height, width)
        print("range of pred_img: [", pred_img.min(), ", ", pred_img.max(), "]")

        # Apply scale factor if needed
        if args.scale_factor != 1.0:
            print(f"Applying scale factor: {args.scale_factor}")
            pred_img = pred_img * args.scale_factor
        
        pred_img = pred_img.clamp(0.0, 255.0)
        # Convert to uint8 range [0, 255] for PNG
        out_img = pred_img.detach().float().cpu().numpy()
        out_img = out_img.astype(np.uint8)

        # Save as PNG
        checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
        out_path = os.path.join(args.out_dir, f"{checkpoint_name}_output.png")
        cv2.imwrite(out_path, out_img)
        print(f"\nSaved output: {out_path}")
        print(f"  Output range: [{out_img.min()}, {out_img.max()}]")
        print(f"  Mean: {out_img.mean():.1f}, Std: {out_img.std():.1f}")
        
        # Compute and visualize variance if requested
        if args.compute_variance:
            print(f"\nComputing variance map...")
            
            # Load PSF covariance
            if not os.path.exists(args.psf_covariance_path):
                print(f"Warning: PSF covariance file not found at {args.psf_covariance_path}")
                print("Skipping variance computation")
            else:
                sigma = torch.load(args.psf_covariance_path).float().to(device)
                print(f"Loaded PSF covariance with shape: {sigma.shape}")
                
                if sigma.shape[0] != 2 or sigma.shape[1] != 2:
                    print(f"Warning: PSF covariance shape {sigma.shape} is not 2x2, skipping variance computation")
                else:
                    psf = gaussian_psf_from_cov(sigma)      # Sigma is your 2x2 covariance in pixel units
                    var_map = variance_from_gt_psf(
                        img, psf, 
                        device=device,
                        tile_size=args.variance_tile_size,
                        overlap=args.variance_overlap
                    )

                    # Save variance and std dev maps
                    std_map = np.sqrt(var_map)
                    
                    # Save as NPY files (full resolution)
                    var_npy_path = os.path.join(args.out_dir, f"{checkpoint_name}_gt_variance.npy")
                    std_npy_path = os.path.join(args.out_dir, f"{checkpoint_name}_gt_std.npy")
                    np.save(var_npy_path, var_map)
                    np.save(std_npy_path, std_map)
                    print(f"Saved GT variance map: {var_npy_path}")
                    print(f"Saved GT std dev map: {std_npy_path}")
                    
                    # Save statistics
                    print(f"\nGT Variance statistics:")
                    print(f"  Min: {var_map.min():.6f}")
                    print(f"  Max: {var_map.max():.6f}")
                    print(f"  Mean: {var_map.mean():.6f}")
                    print(f"  Std: {var_map.std():.6f}")
                    
                    # Save as PNG (normalize to uint8)
                    # Normalize std_map to [0, 255]
                    std_min, std_max = std_map.min(), std_map.max()
                    if std_max > std_min:
                        std_normalized = ((std_map - std_min) / (std_max - std_min) * 255).astype(np.uint8)
                    else:
                        std_normalized = np.zeros_like(std_map, dtype=np.uint8)
                    
                    std_png_path = os.path.join(args.out_dir, f"{checkpoint_name}_gt_std.png")
                    cv2.imwrite(std_png_path, std_normalized)
                    print(f"Saved GT std dev PNG: {std_png_path}")
                    
                    # Save model variance if available
                    if model_var_flat is not None:
                        model_var_img = torch.exp(model_var_flat).view(height, width).cpu().numpy()
                        model_std_img = np.sqrt(model_var_img)
                        
                        # Save as NPY
                        model_var_npy_path = os.path.join(args.out_dir, f"{checkpoint_name}_model_variance.npy")
                        model_std_npy_path = os.path.join(args.out_dir, f"{checkpoint_name}_model_std.npy")
                        np.save(model_var_npy_path, model_var_img)
                        np.save(model_std_npy_path, model_std_img)
                        print(f"Saved model variance map: {model_var_npy_path}")
                        print(f"Saved model std dev map: {model_std_npy_path}")
                        
                        # Save as PNG (normalize to uint8)
                        model_std_min, model_std_max = model_std_img.min(), model_std_img.max()
                        if model_std_max > model_std_min:
                            model_std_normalized = ((model_std_img - model_std_min) / (model_std_max - model_std_min) * 255).astype(np.uint8)
                        else:
                            model_std_normalized = np.zeros_like(model_std_img, dtype=np.uint8)
                        
                        model_std_png_path = os.path.join(args.out_dir, f"{checkpoint_name}_model_std.png")
                        cv2.imwrite(model_std_png_path, model_std_normalized)
                        print(f"Saved model std dev PNG: {model_std_png_path}")
                        
                        print(f"\nModel Variance statistics:")
                        print(f"  Min: {model_var_img.min():.6f}")
                        print(f"  Max: {model_var_img.max():.6f}")
                        print(f"  Mean: {model_var_img.mean():.6f}")
                        print(f"  Std: {model_var_img.std():.6f}")
                    else:
                        print("Model variance not available (model not trained with variance)")


if __name__ == "__main__":
    main()
