"""
SIREN-based PSF fitting for continuous sampling.

This module provides a SIREN MLP (sin activations with special initialization)
to fit a discrete PSF and enable continuous coordinate-based sampling.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class SineLayer(nn.Module):
    """
    SIREN layer with sin activation and special initialization.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        is_first: Whether this is the first layer (uses omega_0)
        omega_0: Frequency multiplier for first layer
        omega: Frequency multiplier for hidden layers
    """
    def __init__(self, in_features: int, out_features: int, 
                 is_first: bool = False, omega_0: float = 30.0, omega: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0 if is_first else omega
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SoftplusLayer(nn.Module):
    """Softplus layer (kept for backward compatibility)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(x))


class SoftplusPSF(nn.Module):
    """
    SIREN-based PSF model for continuous coordinate sampling.
    
    Uses sin activations with SIREN initialization for better high-frequency fitting.
    
    Args:
        n_dims: Number of input dimensions (2 for 2D PSF, 3 for 3D PSF)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        omega_0: Frequency for first layer (default: 30)
        hidden_omega: Frequency for hidden layers (default: 30)
    """
    def __init__(
        self, 
        n_dims: int = 2,
        hidden_dim: int = 64, 
        num_layers: int = 2,
        omega_0: float = 30.0,
        hidden_omega: float = 30.0,
    ):
        super().__init__()
        
        self.n_dims = n_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega_0 = omega_0
        self.hidden_omega = hidden_omega
        
        # Build SIREN network
        layers = []
        
        # First layer with omega_0
        layers.append(SineLayer(n_dims, hidden_dim, is_first=True, 
                               omega_0=omega_0, omega=hidden_omega))
        
        # Hidden layers with hidden_omega
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False,
                                   omega_0=omega_0, omega=hidden_omega))
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer (linear, then softplus for non-negative output)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # SIREN-style initialization for output layer
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / hidden_omega
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.layers(coords)
        # Softplus ensures non-negative output (PSF must be >= 0)
        return F.softplus(self.output_layer(x))
    
    def sample(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Sample PSF values at given coordinates.
        
        Args:
            coords: Normalized coordinates in [-1, 1], shape [N, n_dims] or [..., n_dims]
        
        Returns:
            PSF values, shape [N] or [...]
        """
        original_shape = coords.shape[:-1]
        coords_flat = coords.view(-1, self.n_dims)
        
        with torch.no_grad():
            values = self.forward(coords_flat).squeeze(-1)
        
        return values.view(original_shape)


def create_coord_grid(shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Create a normalized coordinate grid for the given shape.
    
    Args:
        shape: Shape of the grid (H, W) for 2D or (D, H, W) for 3D
        device: Device to create the tensor on
    
    Returns:
        Coordinate tensor of shape (*shape, n_dims) with values in [-1, 1]
    """
    n_dims = len(shape)
    
    # Create 1D coordinates for each dimension, normalized to [-1, 1]
    coords_1d = [torch.linspace(-1, 1, s, device=device) for s in shape]
    
    # Create meshgrid
    grids = torch.meshgrid(*coords_1d, indexing='ij')
    
    # Stack into coordinate tensor
    coords = torch.stack(grids, dim=-1)
    
    return coords


def fit_psf(
    psf: torch.Tensor,
    model: SoftplusPSF,
    device: torch.device,
    steps: int = 1000,
    lr: float = 1e-4,
    batch_size: int = 10000,
    tensorboard_dir: str = None,
) -> dict:
    """
    Fit the Softplus MLP model to a discrete PSF.
    
    Args:
        psf: Discrete PSF tensor, shape (H, W) for 2D or (D, H, W) for 3D
        model: Softplus PSF model
        device: Device to train on
        steps: Number of training steps
        lr: Learning rate
        batch_size: Number of samples per batch
    
    Returns:
        Dictionary with training statistics
    """
    psf = psf.to(device)
    model = model.to(device)
    
    # Store PSF sum for reference (but don't normalize)
    psf_sum = psf.sum()
    
    # Create coordinate grid
    shape = psf.shape
    coords = create_coord_grid(shape, device)  # Shape: (*shape, n_dims)
    coords_flat = coords.view(-1, len(shape))  # Shape: (N, n_dims)
    values_flat = psf.view(-1)  # Shape: (N,) - use raw PSF values
    
    total_pixels = coords_flat.shape[0]
    
    # Precompute foreground indices for biased sampling
    fg_threshold = values_flat.max() * 0.01  # 1% of max as threshold
    fg_indices = torch.nonzero(values_flat > fg_threshold, as_tuple=False).squeeze(-1)
    n_fg = len(fg_indices)
    fg_ratio = 0.2  # 80% foreground, 20% background
    print(f"Foreground pixels: {n_fg:,} / {total_pixels:,} ({100*n_fg/total_pixels:.2f}%)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps.real)
    
    # TensorBoard writer
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logging to: {tensorboard_dir}")
    
    model.train()
    
    pbar = tqdm(range(steps), desc="Fitting Softplus PSF")
    losses = []
    
    for step in pbar:
        # Foreground-biased sampling
        n_fg_samples = int(batch_size * fg_ratio)
        n_bg_samples = batch_size - n_fg_samples
        
        # Sample from foreground
        fg_sample_idx = fg_indices[torch.randint(0, n_fg, (n_fg_samples,), device=device)]
        # Sample uniformly (background)
        bg_sample_idx = torch.randint(0, total_pixels, (n_bg_samples,), device=device)
        
        indices = torch.cat([fg_sample_idx, bg_sample_idx])
        batch_coords = coords_flat[indices]
        batch_values = values_flat[indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch_coords).squeeze(-1)
        
        # MSE loss
        loss = F.mse_loss(pred, batch_values) 
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        # TensorBoard logging
        if writer is not None and step % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)
            
            # Log gradient norms
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    writer.add_scalar(f'gradients/{name}', param_grad_norm, step)
                    total_grad_norm += param_grad_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar('gradients/total_norm', total_grad_norm, step)
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
    
    # Compute final metrics
    model.eval()
    with torch.no_grad():
        # Chunked inference to avoid OOM
        chunk_size = 100000
        full_pred = []
        for start in range(0, coords_flat.shape[0], chunk_size):
            end = min(start + chunk_size, coords_flat.shape[0])
            pred_chunk = model(coords_flat[start:end]).squeeze(-1)
            full_pred.append(pred_chunk)
        full_pred = torch.cat(full_pred, dim=0)
        final_loss = F.mse_loss(full_pred, values_flat).item()
        
        # Compute PSNR
        mse = final_loss
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # Log final metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('eval/final_loss', final_loss, steps)
        writer.add_scalar('eval/psnr', psnr, steps)
        writer.close()
    
    return {
        'final_loss': final_loss,
        'psnr': psnr,
        'losses': losses,
        'psf_sum': psf_sum.item(),
    }


def visualize_psf_comparison(
    psf: torch.Tensor,
    model: SoftplusPSF,
    device: torch.device,
    num_slices: int = 5,
    save_path: str = None,
) -> None:
    """
    Visualize comparison between original discrete PSF and MLP-fitted PSF at different Z slices.
    
    Args:
        psf: Original discrete PSF tensor, shape (D, H, W) for 3D
        model: Trained Softplus PSF model
        device: Device to run inference on
        num_slices: Number of Z slices to visualize
        save_path: Optional path to save the visualization
    """
    model.eval()
    psf = psf.to(device)
    
    # Use normalized PSF values (same as during training)
    
    n_dims = len(psf.shape)
    
    if n_dims == 3:
        # 3D PSF - visualize different Z slices
        D, H, W = psf.shape
        
        # Select evenly spaced slices
        slice_indices = np.linspace(0, D - 1, num_slices, dtype=int)
        
        # Create coordinate grid for the entire volume
        coords = create_coord_grid(psf.shape, device)
        coords_flat = coords.view(-1, n_dims)
        
        # Get MLP predictions for entire volume (chunked to avoid OOM)
        with torch.no_grad():
            chunk_size = 100000
            pred_chunks = []
            for start in range(0, coords_flat.shape[0], chunk_size):
                end = min(start + chunk_size, coords_flat.shape[0])
                pred_chunk = model(coords_flat[start:end]).squeeze(-1)
                pred_chunks.append(pred_chunk)
            pred_flat = torch.cat(pred_chunks, dim=0)
        pred_volume = pred_flat.view(D, H, W)
        
        # Create figure
        fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4 * num_slices))
        
        for i, z_idx in enumerate(slice_indices):
            # Original slice (raw values, no normalization)
            original_slice = psf[z_idx].cpu().numpy()
            print("original range", np.min(original_slice), np.max(original_slice))
            
            # MLP prediction slice (trained on raw PSF values)
            pred_slice = pred_volume[z_idx].cpu().numpy()
            print("pred range", np.min(pred_slice), np.max(pred_slice))
            
            # Debug: print center line profile stats
            center_y = H // 2
            orig_line = original_slice[center_y, :]
            pred_line = pred_slice[center_y, :]
            print(f"  Center line (y={center_y}): orig range [{orig_line.min():.4f}, {orig_line.max():.4f}], pred range [{pred_line.min():.4f}, {pred_line.max():.4f}]")
            print(f"  Line MSE: {np.mean((orig_line - pred_line)**2):.6f}, Max diff: {np.max(np.abs(orig_line - pred_line)):.4f}")
            
            # Error map
            error = np.abs(original_slice - pred_slice)
            
            # Plot original
            im0 = axes[i, 0].imshow(original_slice, cmap='hot', interpolation='nearest')
            axes[i, 0].set_title(f'Original PSF (Z={z_idx}/{D-1})')
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            # Plot MLP prediction
            im1 = axes[i, 1].imshow(pred_slice, cmap='hot', interpolation='nearest')
            axes[i, 1].set_title(f'MLP Fitted PSF (Z={z_idx}/{D-1})')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # Plot error
            im2 = axes[i, 2].imshow(error, cmap='viridis', interpolation='nearest')
            axes[i, 2].set_title(f'Absolute Error (Z={z_idx}/{D-1})')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            # Plot center line profile comparison
            center_y = H // 2
            axes[i, 3].plot(original_slice[center_y, :], 'b-', label='Original', linewidth=2)
            axes[i, 3].plot(pred_slice[center_y, :], 'r--', label='MLP Fitted', linewidth=2)
            axes[i, 3].set_title(f'Center Line Profile (Z={z_idx}/{D-1})')
            axes[i, 3].set_xlabel('X')
            axes[i, 3].set_ylabel('Intensity')
            axes[i, 3].legend()
            axes[i, 3].grid(True, alpha=0.3)
            
            # Compute metrics for this slice
            mse = np.mean((original_slice - pred_slice) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            axes[i, 3].text(0.02, 0.98, f'MSE: {mse:.6f}\nPSNR: {psnr:.2f} dB', 
                           transform=axes[i, 3].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
    elif n_dims == 2:
        # 2D PSF - just show one comparison
        H, W = psf.shape
        
        # Create coordinate grid
        coords = create_coord_grid(psf.shape, device)
        coords_flat = coords.view(-1, n_dims)
        
        # Get MLP predictions
        with torch.no_grad():
            pred_flat = model(coords_flat).squeeze(-1)
        pred_image = pred_flat.view(H, W)
        
        original_image = psf.cpu().numpy()
        pred_image = pred_image.cpu().numpy()
        error = np.abs(original_image - pred_image)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot original
        im0 = axes[0].imshow(original_image, cmap='hot', interpolation='nearest')
        axes[0].set_title('Original PSF')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Plot MLP prediction
        im1 = axes[1].imshow(pred_image, cmap='hot', interpolation='nearest')
        axes[1].set_title('MLP Fitted PSF')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Plot error
        im2 = axes[2].imshow(error, cmap='viridis', interpolation='nearest')
        axes[2].set_title('Absolute Error')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # Plot center line profile comparison
        center_y = H // 2
        axes[3].plot(original_image[center_y, :], 'b-', label='Original', linewidth=2)
        axes[3].plot(pred_image[center_y, :], 'r--', label='MLP Fitted', linewidth=2)
        axes[3].set_title('Center Line Profile')
        axes[3].set_xlabel('X')
        axes[3].set_ylabel('Intensity')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Compute metrics
        mse = np.mean((original_image - pred_image) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        axes[3].text(0.02, 0.98, f'MSE: {mse:.6f}\nPSNR: {psnr:.2f} dB', 
                    transform=axes[3].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fit a Softplus MLP to a discrete PSF for continuous sampling")
    
    # Input/output
    parser.add_argument("--psf_path", type=str, default="/workspace/FLFM/LFPSF_H_20260105_222810.tif",
                        help="Path to discrete PSF file (.tif or .npy)")
    parser.add_argument("--output_path", type=str, default="../checkpoints/lfm_psf.pth",
                        help="Path to save the trained Softplus model")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension of the SIREN MLP")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="Number of hidden layers")
    parser.add_argument("--omega_0", type=float, default=30.0,
                        help="Frequency multiplier for first layer")
    parser.add_argument("--hidden_omega", type=float, default=30.0,
                        help="Frequency multiplier for hidden layers")
    
    # Training
    parser.add_argument("--steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100000,
                        help="Batch size for training")
    parser.add_argument("--num_viz_slices", type=int, default=5,
                        help="Number of Z slices to visualize in comparison plot")
    parser.add_argument("--tensorboard_dir", type=str, default="../runs/lfm_psf",
                        help="Directory for TensorBoard logs (optional)")
    
    args = parser.parse_args()
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PSF
    print(f"Loading PSF from {args.psf_path}")
    if args.psf_path.endswith(".tif") or args.psf_path.endswith(".tiff"):
        psf_np = tifffile.imread(args.psf_path).astype(np.float32)
        psf_np = psf_np / 65535.0
    elif args.psf_path.endswith(".npy"):
        psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {args.psf_path}")
    
    psf = torch.from_numpy(psf_np)
    n_dims = len(psf.shape)
    
    print(f"PSF shape: {psf.shape}")
    print(f"PSF range: [{psf.min():.6f}, {psf.max():.6f}]")
    print(f"PSF dimensions: {n_dims}D")
    
    # Create model
    model = SoftplusPSF(
        n_dims=n_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
        hidden_omega=args.hidden_omega,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Softplus model parameters: {total_params:,}")
    
    # Fit model
    print(f"\nTraining for {args.steps} steps...")
    stats = fit_psf(
        psf=psf,
        model=model,
        device=device,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        tensorboard_dir=args.tensorboard_dir,
    )
    
    print(f"\nTraining complete!")
    print(f"  Final MSE: {stats['final_loss']:.6f}")
    print(f"  PSNR: {stats['psnr']:.2f} dB")
    
    # Save model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'n_dims': n_dims,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'omega_0': args.omega_0,
        'hidden_omega': args.hidden_omega,
        'psf_shape': list(psf.shape),
        'psf_sum': stats['psf_sum'],
        'final_loss': stats['final_loss'],
        'psnr': stats['psnr'],
    }
    
    torch.save(save_dict, args.output_path)
    print(f"Model saved to {args.output_path}")
    
    # Demo: sample at continuous coordinates
    print("\n--- Demo: Continuous Sampling ---")
    model.eval()
    
    # Sample at center
    center_coord = torch.zeros(1, n_dims, device=device)
    center_value = model.sample(center_coord)
    print(f"Value at center (0, 0, ...): {center_value.item():.6f}")
    
    # Sample at a sub-pixel offset
    offset_coord = torch.tensor([[0.01] * n_dims], device=device)
    offset_value = model.sample(offset_coord)
    print(f"Value at slight offset (0.01, 0.01, ...): {offset_value.item():.6f}")
    
    # Visualize comparison
    print("\n--- Generating Visualization ---")
    viz_save_path = args.output_path.replace('.pth', '_comparison.png')
    visualize_psf_comparison(
        psf=psf,
        model=model,
        device=device,
        num_slices=args.num_viz_slices,
        save_path=viz_save_path,
    )



def load_siren_psf(checkpoint_path: str, device: torch.device = None) -> SoftplusPSF:
    """
    Load a trained Softplus PSF model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device to load the model on (default: cuda if available)
    
    Returns:
        Loaded SoftplusPSF model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = SoftplusPSF(
        n_dims=checkpoint['n_dims'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        omega_0=checkpoint['omega_0'],
        hidden_omega=checkpoint['hidden_omega'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    main()
