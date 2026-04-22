import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import trimesh
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from sample_strategy.random_sample import sample_random_coords
from sample_strategy.distribution_sample import sample_importance_coords
from sample_strategy.sdf_sample import sample_sdf_coords
from regularization.TVs import compute_hyperlaplacian_tv

from visualize.utils import build_grid, gaussian_psf_from_cov_torch
from debug.adaptive import psf_adaptive_sampling_step_debug
from siren_psf import SoftplusPSF, load_siren_psf


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as used in NeRF.
    Maps input coordinates to higher dimensional space using sin/cos functions.
    """
    def __init__(self, n_input_dims=3, n_frequencies=10, include_input=True):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_frequencies = n_frequencies
        self.include_input = include_input
        
        # Compute output dimension
        self.n_output_dims = n_input_dims * n_frequencies * 2
        if include_input:
            self.n_output_dims += n_input_dims
        
        # Create frequency bands (log-linear scaling as in NeRF paper)
        freq_bands = 2.0 ** torch.linspace(0, n_frequencies - 1, n_frequencies)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x):
        """
        x: [N, n_input_dims] coordinates in [0, 1]
        returns: [N, n_output_dims] positional encoded features
        """
        # Scale from [0, 1] to [-1, 1] for better encoding
        x_scaled = x * 2 - 1
        
        # Apply sin and cos at different frequencies
        encodings = []
        if self.include_input:
            encodings.append(x_scaled)
        
        for freq in self.freq_bands:
            encodings.append(torch.sin(np.pi * freq * x_scaled))
            encodings.append(torch.cos(np.pi * freq * x_scaled))
        
        return torch.cat(encodings, dim=-1)


class MLPOnlyModel(nn.Module):
    """
    Pure MLP model with sinusoidal positional encoding.
    No hash grid encoding - uses only positional encoding + MLP.
    """
    def __init__(
        self,
        n_input_dims=3,
        hidden_dim=256,
        num_layers=8,
        n_frequencies=10,
        learn_variance=False,
        skip_connections=[4],  # Layer indices where to inject input (like NeRF)
    ):
        super().__init__()
        
        self.n_input_dims = n_input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learn_variance = learn_variance
        self.skip_connections = skip_connections
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            n_input_dims=n_input_dims,
            n_frequencies=n_frequencies,
            include_input=True
        )
        
        encoding_dim = self.positional_encoding.n_output_dims
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        
        in_dim = encoding_dim
        for i in range(num_layers):
            # Add skip connection input dimension at specified layers
            if i in skip_connections:
                in_dim = in_dim + encoding_dim
            
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        # Output layer
        out_dim = 2 if learn_variance else 1
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Store for dummy progressive training compatibility
        self.current_max_level = num_layers
        self.n_levels = num_layers
        
        print(f"MLPOnlyModel initialized:")
        print(f"  Input dims: {n_input_dims}")
        print(f"  Positional encoding frequencies: {n_frequencies}")
        print(f"  Encoding output dim: {encoding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Skip connections at layers: {skip_connections}")
        print(f"  Learn variance: {learn_variance}")
    
    def set_max_level(self, max_level):
        """Dummy method for compatibility with progressive training code"""
        self.current_max_level = min(max_level, self.n_levels)
    
    def forward(self, x, variance=None, stochastic_alpha=None):
        """
        x: [N, n_input_dims] coordinates in [0, 1]
        returns: [N] density values
        """
        coords = x
        
        # Stochastic preconditioning (same as original)
        if stochastic_alpha is not None and stochastic_alpha > 0:
            noise = stochastic_alpha * torch.normal(
                torch.zeros_like(coords), torch.ones_like(coords)
            )
            coords = coords + noise
            
            # Reflect around boundary
            coords = coords % 2
            mask = coords > 1
            coords[mask] = 2 - coords[mask]
        
        # Positional encoding
        encoded = self.positional_encoding(coords)
        encoded_input = encoded  # Save for skip connections
        
        # MLP forward pass
        h = encoded
        for i, layer in enumerate(self.layers):
            # Apply skip connection
            if i in self.skip_connections:
                h = torch.cat([h, encoded_input], dim=-1)
            
            h = layer(h)
            h = F.relu(h)
        
        # Output
        output = self.output_layer(h)
        
        # Apply Softplus to ensure output > 0
        output = F.softplus(output)
        
        if self.learn_variance:
            return output[..., 0], output[..., 1]
        
        return output.squeeze(-1), None


class VolumeDataset(Dataset):
    def __init__(self, norm_volume_tensor, num_mc_samples, num_pixels_per_step, num_batches, clear_volume_tensor=None, gt_variance_tensor=None, psf_type='siren', discrete_psf=None, siren_psf_model=None, psf_shape=None, mesh=None, importance_map=None, importance_ratio=0.8):
        self.norm_volume_tensor = norm_volume_tensor
        self.clear_volume_tensor = clear_volume_tensor
        self.gt_variance_tensor = gt_variance_tensor
        self.num_mc_samples = num_mc_samples
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.volume_shape = norm_volume_tensor.shape
        self.psf_type = psf_type
        self.discrete_psf = discrete_psf.cpu() if discrete_psf is not None else None
        self.siren_psf_model = siren_psf_model  # SIREN PSF model for continuous sampling
        self.psf_shape = psf_shape  # Shape of the PSF for SIREN sampling
        self.mesh = mesh
        self.importance_map = importance_map
        self.importance_ratio = importance_ratio
        
        # Handle both 2D and 3D
        if len(self.volume_shape) == 3:
            vz, vy, vx = self.volume_shape
            self.inv_shape = torch.tensor([1.0 / (vz - 1), 1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)
        else:
            vy, vx = self.volume_shape
            self.inv_shape = torch.tensor([1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)
        
    def __len__(self):
        return self.num_batches
    
    def _generate_offsets(self):
        sampling_budget = self.num_mc_samples * self.num_pixels_per_step
        n_dims = len(self.volume_shape)
        
        if self.psf_type == 'siren':
            # SIREN-based continuous sampling
            psf_shape = self.psf_shape
            
            # Generate uniform random coordinates in [-1, 1] - truly continuous
            siren_coords = torch.rand(sampling_budget, n_dims) * 2 - 1  # [-1, 1]
            
            # Convert from [-1, 1] to pixel offsets
            if n_dims == 3:
                d, h, w = psf_shape
                z_offset = siren_coords[:, 0] * (d - 1) / 2.0
                y_offset = siren_coords[:, 1] * (h - 1) / 2.0
                x_offset = siren_coords[:, 2] * (w - 1) / 2.0
                offsets = torch.stack([z_offset, y_offset, x_offset], dim=1)
            else:
                h, w = psf_shape
                y_offset = siren_coords[:, 0] * (h - 1) / 2.0
                x_offset = siren_coords[:, 1] * (w - 1) / 2.0
                offsets = torch.stack([y_offset, x_offset], dim=1)
            
            return offsets, siren_coords
            
        elif self.psf_type == 'discrete':
            # Sample from discrete PSF
            psf_shape = self.discrete_psf.shape
            
            # Flatten PSF and compute probabilities
            psf_flat = self.discrete_psf.flatten()
            psf_flat = psf_flat / psf_flat.sum()  # Normalize to probabilities
            
            # Sample indices from the discrete PSF distribution
            sampled_indices = torch.multinomial(psf_flat, sampling_budget, replacement=True)
            
            # Convert flat indices to coordinates
            if n_dims == 3:
                d, h, w = psf_shape
                z_idx = sampled_indices // (h * w)
                y_idx = (sampled_indices % (h * w)) // w
                x_idx = sampled_indices % w
                # Convert to offsets centered around PSF center
                z_offset = z_idx.float() - (d - 1) / 2.0
                y_offset = y_idx.float() - (h - 1) / 2.0
                x_offset = x_idx.float() - (w - 1) / 2.0
                offsets = torch.stack([z_offset, y_offset, x_offset], dim=1)
            else:
                h, w = psf_shape
                y_idx = sampled_indices // w
                x_idx = sampled_indices % w
                # Convert to offsets centered around PSF center
                y_offset = y_idx.float() - (h - 1) / 2.0
                x_offset = x_idx.float() - (w - 1) / 2.0
                offsets = torch.stack([y_offset, x_offset], dim=1)
            
            return offsets
    
    def _sample_sdf_coords(self, n):
        return sample_sdf_coords(
            mesh=self.mesh,
            n=n,
            volume_shape=self.volume_shape,
            device='cpu'
        )
    
    def __getitem__(self, idx):
        if self.mesh is not None and len(self.volume_shape) == 3:
            target_coords = self._sample_sdf_coords(self.num_pixels_per_step)
        elif self.importance_map is not None:
            n_imp = int(self.num_pixels_per_step * self.importance_ratio)
            n_uni = self.num_pixels_per_step - n_imp

            coords_imp = sample_importance_coords(self.importance_map, n_imp, self.volume_shape, tile_size=16, num_tiles=35)
            coords_uni = sample_random_coords(n_uni, self.volume_shape)
            target_coords = torch.cat([coords_imp, coords_uni], dim=0)
        else:
            target_coords = sample_random_coords(self.num_pixels_per_step, self.volume_shape)
        
        # Get target values
        if len(self.volume_shape) == 3:
            z_indices = target_coords[:, 0]
            y_indices = target_coords[:, 1]
            x_indices = target_coords[:, 2]
            target_values = self.norm_volume_tensor[z_indices, y_indices, x_indices]
            clear_values = self.clear_volume_tensor[z_indices, y_indices, x_indices] if self.clear_volume_tensor is not None else None
            gt_variance_values = self.gt_variance_tensor[z_indices, y_indices, x_indices] if self.gt_variance_tensor is not None else None
        else:
            y_indices = target_coords[:, 0]
            x_indices = target_coords[:, 1]
            target_values = self.norm_volume_tensor[y_indices, x_indices]
            clear_values = self.clear_volume_tensor[y_indices, x_indices] if self.clear_volume_tensor is not None else None
            gt_variance_values = self.gt_variance_tensor[y_indices, x_indices] if self.gt_variance_tensor is not None else None
        
        # Generate fresh offsets on every iteration (only if PSF is used)
        offsets_result = self._generate_offsets()
        
        # Handle SIREN which returns tuple (offsets, siren_coords)
        if self.psf_type == 'siren':
            sampled_offsets, siren_coords = offsets_result
        else:
            sampled_offsets = offsets_result
            siren_coords = None
        
        target_coords_normalized = target_coords.float() * self.inv_shape
        
        batch_dict = {
            'target_coords': target_coords_normalized,
            'target_values': target_values.float(),
            'psf_type': self.psf_type,
        }
        
        sampled_offsets_normalized = sampled_offsets.float() * self.inv_shape
        batch_dict['sampled_offsets'] = sampled_offsets_normalized
        
        # Include siren_coords for PSF weight computation in training step
        if siren_coords is not None:
            batch_dict['siren_coords'] = siren_coords.float()
        
        if clear_values is not None:
            batch_dict['clear_values'] = clear_values.float()
        
        if gt_variance_values is not None:
            batch_dict['gt_variance_values'] = gt_variance_values.float()
            
        return batch_dict


def psf_uniform_sampling_step(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
    tv_loss_weight: float = 1e-4,
    stochastic_alpha: float = 0.0,
    pixel_samples: int = 1,  # Anti-aliasing: samples per pixel
    pixel_scale: torch.Tensor = None,  # For jittering within pixel
    siren_psf_model: nn.Module = None,  # SIREN PSF model for weighted averaging
) -> torch.Tensor:
    # Get pre-computed data from DataLoader
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets = batch['sampled_offsets'].to(device, non_blocking=True)
    psf_type = batch.get('psf_type', 'discrete')
    
    # Handle string in batch (from DataLoader it might be a list)
    if isinstance(psf_type, list):
        psf_type = psf_type[0]
    
    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    sampled_offsets = sampled_offsets.view(num_pixels, num_mc_samples, n_dims)
    
    # Get siren_coords if using SIREN PSF
    siren_coords = None
    if psf_type == 'siren' and 'siren_coords' in batch:
        siren_coords = batch['siren_coords'].to(device, non_blocking=True)
        siren_coords = siren_coords.view(num_pixels, num_mc_samples, n_dims)
    
    # Anti-aliased integration: sample multiple points within each pixel
    if pixel_samples > 1 and pixel_scale is not None:
        pixel_jitter = (torch.rand(num_pixels, num_mc_samples, pixel_samples, n_dims, device=device) - 0.5) * pixel_scale.to(device)
        
        source_coords = target_coords.unsqueeze(1) + sampled_offsets
        source_coords = source_coords.unsqueeze(2) + pixel_jitter
        source_coords = torch.clamp(source_coords, 0.0, 1.0)
        source_coords_flat = source_coords.view(-1, n_dims)
    else:
        source_coords = target_coords.unsqueeze(1) + sampled_offsets
        source_coords = torch.clamp(source_coords, 0.0, 1.0)
        source_coords_flat = source_coords.view(-1, n_dims)

    # Reorder coordinates for model input
    if n_dims == 3:
        # 3D: reorder to x, y, z
        coords_for_model = torch.stack([
            source_coords_flat[:, 2],  # x
            source_coords_flat[:, 1],  # y
            source_coords_flat[:, 0]   # z
        ], dim=-1)
    else:
        # 2D: reorder to x, y
        coords_for_model = torch.stack([
            source_coords_flat[:, 1],  # x
            source_coords_flat[:, 0]   # y
        ], dim=-1)
    
    coords_for_model = coords_for_model.float()
    coords_for_model.requires_grad_(False)
    
    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    
    # Reshape and average predictions
    if pixel_samples > 1 and pixel_scale is not None:
        pred_samples = pred_flat.view(num_pixels, num_mc_samples, pixel_samples)
        pred_samples = pred_samples.mean(dim=2)  # Average over pixel samples (anti-aliasing)
    else:
        pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    
    # Simulate observed values by averaging over PSF samples
    if psf_type == 'siren' and siren_psf_model is not None and siren_coords is not None:
        # Compute PSF weights from SIREN model
        siren_coords_flat = siren_coords.view(-1, n_dims).to(device)
        with torch.no_grad():
            psf_weights = siren_psf_model(siren_coords_flat)
            psf_weights = psf_weights.squeeze(-1)
            psf_weights = torch.clamp(psf_weights, min=0)
        psf_weights = psf_weights.view(num_pixels, num_mc_samples)
        
        # Normalize weights per pixel
        psf_weights = psf_weights / (psf_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted average
        simulated_values = (pred_samples * psf_weights).sum(dim=1)
    else:
        # Simple mean for discrete sampling
        simulated_values = pred_samples.mean(dim=1)
    
    total_loss = 0.0
    data_loss_mc = F.mse_loss(simulated_values.float(), target_values.float())

    loss_dict = {
        "reconstruction_loss": data_loss_mc,
    }
    total_loss += data_loss_mc

    
    if tv_loss_weight is not None:
        coords_grid = build_grid(n_dims, 64, device)
        tv_loss = compute_hyperlaplacian_tv(model, coords_grid).mean()
        total_loss += tv_loss_weight * tv_loss
        loss_dict["tv_loss"] = tv_loss * tv_loss_weight

    loss_dict["total_loss"] = total_loss
    
    return loss_dict


def simple_train_step(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    pixel_samples: int = 1,
    pixel_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Simple training step with direct MSE loss (no PSF)
    """
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    
    n_dims = target_coords.shape[1]
    num_pixels = target_coords.shape[0]
    
    # Anti-aliased integration
    if pixel_samples > 1 and pixel_scale is not None:
        pixel_jitter = (torch.rand(num_pixels, pixel_samples, n_dims, device=device) - 0.5) * pixel_scale.to(device)
        coords_expanded = target_coords.unsqueeze(1) + pixel_jitter
        coords_expanded = torch.clamp(coords_expanded, 0.0, 1.0)
        coords_flat = coords_expanded.view(-1, n_dims)
    else:
        coords_flat = target_coords
    
    # Reorder coordinates for model input
    if n_dims == 3:
        coords_for_model = torch.stack([
            coords_flat[:, 2],  # x
            coords_flat[:, 1],  # y
            coords_flat[:, 0]   # z
        ], dim=-1)
    else:
        coords_for_model = torch.stack([
            coords_flat[:, 1],  # x
            coords_flat[:, 0]   # y
        ], dim=-1)
    
    # Forward pass
    predictions, _ = model(coords_for_model, variance=None)
    
    # Average over pixel samples if applicable
    if pixel_samples > 1 and pixel_scale is not None:
        predictions = predictions.view(num_pixels, pixel_samples).mean(dim=1)
    
    # Compute MSE loss
    loss = F.mse_loss(predictions.float(), target_values.float())
    
    return {"reconstruction_loss": loss, "total_loss": loss}


def psf_adaptive_sampling_step(
    model: nn.Module,
    batch: dict,
    volume_shape: tuple,
    num_mc_samples: int,
    device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Adaptive sampling step using model-predicted variance
    """
    target_coords_normalized = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets_normalized = batch['sampled_offsets'].to(device, non_blocking=True)

    n_dims = target_coords_normalized.shape[1]

    if n_dims == 3:
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 2],
            target_coords_normalized[:, 1],
            target_coords_normalized[:, 0]
        ], dim=-1)
    else:
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 1],
            target_coords_normalized[:, 0]
        ], dim=-1)

    _, log_variance = model(target_coords_for_model)
    log_variance = log_variance.view(-1)
    variance = torch.exp(log_variance)

    # Compute adaptive MC budget per point
    sampling_budget = num_mc_samples * target_coords_normalized.shape[0]
    num_samples_per_data = sampling_budget * variance / (variance.sum() + eps)
    num_samples_per_data_floor = num_samples_per_data.floor()
    missing_samples = num_samples_per_data - num_samples_per_data_floor
    num_samples_per_data_floor = num_samples_per_data_floor.long()

    # At least 1 sample per data
    num_samples_per_data_floor = torch.clamp(num_samples_per_data_floor, min=1)

    total_samples = num_samples_per_data_floor.sum().item()
    budget_diff = sampling_budget - total_samples

    abs_budget_diff = abs(budget_diff)
    adjustment_sign = 1 if budget_diff > 0 else -1

    selection_values = missing_samples if budget_diff > 0 else num_samples_per_data_floor.float()
    abs_budget_diff = int(min(abs_budget_diff, selection_values.shape[0]))

    if abs_budget_diff > 0:
        _, topk_indices = selection_values.topk(abs_budget_diff, dim=-1, sorted=False)
        num_samples_per_data_floor[topk_indices] += adjustment_sign

    num_samples_per_data = num_samples_per_data_floor

    # Resample coords with offsets (MC)
    x_resampled_normalized = torch.repeat_interleave(
        target_coords_normalized, num_samples_per_data, dim=0
    ) + sampled_offsets_normalized

    x_resampled_normalized = torch.clamp(x_resampled_normalized, 0.0, 1.0)

    if n_dims == 3:
        x_resampled_for_model = torch.stack([
            x_resampled_normalized[:, 2],
            x_resampled_normalized[:, 1],
            x_resampled_normalized[:, 0]
        ], dim=-1)
    else:
        x_resampled_for_model = torch.stack([
            x_resampled_normalized[:, 1],
            x_resampled_normalized[:, 0]
        ], dim=-1)

    # Model prediction on MC samples
    pred_flat, _ = model(x_resampled_for_model)
    pred_flat = pred_flat.view(-1)

    # Indices: MC sample -> base point index
    indices = torch.arange(
        target_coords_normalized.shape[0],
        device=target_coords_normalized.device
    )
    indices_resampled = torch.repeat_interleave(indices, num_samples_per_data, dim=0)

    # Reconstruction loss: average MC samples per point
    pred_target_mc = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    pred_target_mc.scatter_add_(0, indices_resampled, pred_flat)
    pred_target_mc = pred_target_mc / num_samples_per_data.float()

    loss_reconstruction = F.mse_loss(pred_target_mc, target_values.float())

    # Variance supervision
    E_x2 = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    E_x2.scatter_add_(0, indices_resampled, pred_flat ** 2)
    E_x2 = E_x2 / num_samples_per_data.float()

    E_x = target_values.float()

    var_gt = E_x2 - E_x ** 2

    pred_var = variance.view_as(var_gt)

    loss_variance = F.mse_loss(pred_var, var_gt)

    # Total loss
    total_loss = loss_reconstruction + loss_variance

    return {
        "reconstruction_loss": loss_reconstruction,
        "variance_loss": loss_variance,
        "total_loss": total_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="/workspace/LSM/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--psf_path", type=str, default="/workspace/LSM/psf_t0_v0.tif")
    parser.add_argument("--psf_type", type=str, default="discrete", choices=["discrete", "siren"], help="Type of PSF: 'discrete' for discrete PSF (.npy/.tif), or 'siren' for continuous SIREN model (.pth)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower LR for MLP
    parser.add_argument("--save_path", type=str, default="../checkpoints/mlp_only.pth")
    parser.add_argument("--logdir", type=str, default="../runs/mlp_only")
    parser.add_argument("--num_mc_samples", type=int, default=50, help="Number of PSF samples per pixel")
    parser.add_argument("--num_pixels_per_step", type=int, default=180000, help="Number of pixels per step")
    parser.add_argument("--sampling_strategy", type=str, default="uniform", choices=["uniform", "adaptive", "simple"])
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file (.obj, .ply, etc.) for SDF-based sampling (3D only)")
    
    parser.add_argument("--tv_loss_weight", type=float, default=None, help="Weight for total variation loss")
    
    # Stochastic training params
    parser.add_argument("--sp_alpha_init", type=float, default=0.03, 
                        help="Initial std dev for stochastic preconditioning")
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33, 
                        help="Fraction of training steps over which to decay alpha to 0")

    parser.add_argument("--gt_variance_path", type=str, default=None, help="Path to ground truth variance .npy file for debug adaptive sampling")
    
    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Clear image regularization
    parser.add_argument("--clear_image_path", type=str, default=None, help="Path to clear/sharp image for L1 regularization")
    parser.add_argument("--clear_loss_weight", type=float, default=0.1, help="Weight for clear image L1 regularization loss")

    # MLP-only model config (no hash grid)
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of MLP layers")
    parser.add_argument("--n_frequencies", type=int, default=5, help="Number of positional encoding frequencies")
    
    # Anti-aliasing
    parser.add_argument("--pixel_samples", type=int, default=1, help="Number of samples per pixel for anti-aliased training (1=point sampling, 4=recommended for AA)")
    

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image and convert to tensor
    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_np = vol / 65535.0 
    del vol
    vol_norm = torch.from_numpy(vol_np)
    print(f"Loaded volume with shape: {vol_norm.shape}, dtype: {vol_norm.dtype}, range: [{vol_norm.min():.4f}, {vol_norm.max():.4f}]")

    importance_map = None
    
    # Load clear image if provided
    clear_vol_norm = None
    if args.clear_image_path is not None:
        print(f"Loading clear image from {args.clear_image_path}")
        if args.clear_image_path.endswith(".tif"):
            clear_vol = tifffile.imread(args.clear_image_path).astype(np.float32)
        else:
            clear_vol = cv2.imread(args.clear_image_path, cv2.IMREAD_UNCHANGED)
            clear_vol = clear_vol.astype(np.float32)
        
        # Normalize clear image with same factor as blurred image
        clear_vol_norm_np = clear_vol / vol.max()
        clear_vol_norm = torch.from_numpy(clear_vol_norm_np).float()
        
        # Verify dimensions match
        if clear_vol_norm.shape != vol_norm.shape:
            raise ValueError(f"Clear image shape {clear_vol_norm.shape} does not match blurred image shape {vol_norm.shape}")
        
        print(f"Clear image loaded with shape: {clear_vol_norm.shape}, normalized max: {clear_vol_norm.max():.4f}")
    
    # Load ground truth variance if provided (for debug adaptive sampling)
    gt_variance_map = None
    if args.gt_variance_path is not None:
        print(f"Loading ground truth variance from {args.gt_variance_path}")
        gt_variance_np = np.load(args.gt_variance_path).astype(np.float32)
        gt_variance_map = torch.from_numpy(gt_variance_np).float().to(device)
        
        # Verify dimensions match
        if gt_variance_map.shape != vol_norm.shape:
            raise ValueError(f"GT variance shape {gt_variance_map.shape} does not match blurred image shape {vol_norm.shape}")
        
        print(f"GT variance loaded with shape: {gt_variance_map.shape}")
        print(f"GT variance statistics: min={gt_variance_map.min():.6f}, max={gt_variance_map.max():.6f}, mean={gt_variance_map.mean():.6f}")
    
    # Detect 2D or 3D
    n_dims = len(vol_norm.shape)
    
    # Compute pixel scale for anti-aliased training (1 pixel in normalized coords)
    if n_dims == 3:
        pixel_scale = torch.tensor([1.0 / (vol_norm.shape[0] - 1), 
                                     1.0 / (vol_norm.shape[1] - 1), 
                                     1.0 / (vol_norm.shape[2] - 1)])
    else:
        pixel_scale = torch.tensor([1.0 / (vol_norm.shape[0] - 1), 
                                     1.0 / (vol_norm.shape[1] - 1)])
    print(f"Pixel scale for anti-aliasing: {pixel_scale}")
    if args.pixel_samples > 1:
        print(f"Anti-aliased training enabled: {args.pixel_samples} samples per pixel")
    
    # Build model config for saving
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "n_frequencies": args.n_frequencies,
        "n_input_dims": n_dims,
        "learn_variance": (args.sampling_strategy == "adaptive"),
    }
    
    # Load from checkpoint if specified
    if args.load_checkpoint is not None:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        
        # Load model config from checkpoint
        model_config = checkpoint['model_config']
        print("Loaded model config from checkpoint:")
        print(f"  {model_config}")
    
    # Create MLP-only model (no hash grid!)
    model = MLPOnlyModel(
        n_input_dims=n_dims,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        n_frequencies=model_config["n_frequencies"],
        learn_variance=model_config.get("learn_variance", False),
        skip_connections=[4],  # Skip connection at layer 4 (like NeRF)
    ).to(device)
    
    # Load model state dict if checkpoint was provided
    if args.load_checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint")
    
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    
    # Load optimizer state if checkpoint was provided
    if args.load_checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Starting with fresh optimizer state")
    
    gamma = 1000 ** (-1 / args.steps) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Load scheduler state if checkpoint was provided
    if args.load_checkpoint is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            print("Starting with fresh scheduler state")
    
    # GradScaler for FP16 training
    scaler = GradScaler('cuda')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)
    
    # Load PSF (only if not using simple strategy)
    discrete_psf = None
    siren_psf_model = None
    psf_shape = None
    
    if args.sampling_strategy != "simple":
        if args.psf_type == "discrete":
            # Load discrete PSF from .npy or .tif file
            if args.psf_path.endswith(".tif") or args.psf_path.endswith(".tiff"):
                discrete_psf_np = tifffile.imread(args.psf_path).astype(np.float32)
            elif args.psf_path.endswith(".npy"):
                discrete_psf_np = np.load(args.psf_path).astype(np.float32)
            else:
                raise ValueError(f"Unsupported PSF file format: {args.psf_path}. Expected .tif, .tiff, or .npy")
            
            discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
            print(f"Discrete PSF shape: {discrete_psf.shape}")
            print(f"Discrete PSF min: {discrete_psf.min():.6f}, max: {discrete_psf.max():.6f}")
            
            # Verify PSF dimensions match data dimensions
            if len(discrete_psf.shape) != n_dims:
                raise ValueError(f"Discrete PSF dimensions {len(discrete_psf.shape)} does not match data dimensions {n_dims}D")
        
        elif args.psf_type == "siren":
            # Load pre-trained SIREN PSF model for continuous sampling
            print(f"Loading SIREN PSF model from {args.psf_path}")
            siren_psf_model = load_siren_psf(args.psf_path, device=device)
            
            # Load checkpoint to get PSF shape
            checkpoint = torch.load(args.psf_path, map_location='cpu', weights_only=False)
            psf_shape = tuple(checkpoint['psf_shape'])
            
            print(f"SIREN PSF loaded successfully")
            print(f"  PSF shape: {psf_shape}")
            print(f"  Model dims: {checkpoint['n_dims']}D")
            print(f"  Hidden dim: {checkpoint['hidden_dim']}, Layers: {checkpoint['num_layers']}")
            print(f"  Training PSNR: {checkpoint['psnr']:.2f} dB")
            
            # Verify PSF dimensions match data dimensions
            if checkpoint['n_dims'] != n_dims:
                raise ValueError(f"SIREN PSF dimensions {checkpoint['n_dims']} does not match data dimensions {n_dims}D")
            
    else:
        # For simple strategy, PSF not needed
        print("Simple strategy: PSF not loaded")


    # Load mesh for SDF sampling if mesh_path is provided
    mesh = None
    if args.mesh_path is not None:
        print(f"Loading mesh from {args.mesh_path} for SDF-based sampling...")
        mesh = trimesh.load(args.mesh_path)
        print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print("Using SDF-based sampling strategy (bounds=1.2, uniform_ratio=0.25)")
    
    # Create dataset and dataloader for parallel data preparation
    dataset = VolumeDataset(
        norm_volume_tensor=vol_norm,
        num_mc_samples=args.num_mc_samples if args.sampling_strategy != "simple" else 1,
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
        clear_volume_tensor=clear_vol_norm,
        gt_variance_tensor=gt_variance_map.cpu() if gt_variance_map is not None else None,
        psf_type=args.psf_type,
        discrete_psf=discrete_psf.cpu() if discrete_psf is not None else None,
        siren_psf_model=siren_psf_model,
        psf_shape=psf_shape,
        mesh=mesh,
        importance_map=importance_map,    
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=24,  
    )
    
    # Create iterator
    data_iter = iter(dataloader)
    
    # Early stopping configuration
    early_stop_patience = 10000000
    early_stop_min_delta = 1e-6
    best_loss = float('inf')
    patience_counter = 0
    
    # Stochastic preconditioning decay steps 
    sp_decay_steps = int(args.steps * args.sp_decay_fraction)

    pbar = tqdm(
        total=args.steps,
        desc="Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    print(f"Sampling strategy: {args.sampling_strategy}")
    
    if args.sampling_strategy == "adaptive" and gt_variance_map is not None:
        print("=" * 60)
        print("DEBUG MODE: Using ground truth variance for adaptive sampling")
        print("=" * 60)

    
    for step in range(args.steps):
        batch = next(data_iter)
        batch = {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        if step < sp_decay_steps:
            progress = step / sp_decay_steps
            current_alpha = args.sp_alpha_init * np.exp(-5.0 * progress)
        else:
            current_alpha = 0.0
        
        if args.sampling_strategy == "simple":
            # Simple training: direct MSE without PSF
            loss_dict = simple_train_step(
                model=model,
                batch=batch,
                device=device,
                pixel_samples=args.pixel_samples,
                pixel_scale=pixel_scale,
            )
        elif args.sampling_strategy == "uniform":
            loss_dict = psf_uniform_sampling_step(
                model=model,
                batch=batch,
                num_mc_samples=args.num_mc_samples,
                device=device,
                tv_loss_weight=args.tv_loss_weight,
                stochastic_alpha=current_alpha,
                pixel_samples=args.pixel_samples,
                pixel_scale=pixel_scale,
                siren_psf_model=siren_psf_model,
            )
        elif args.sampling_strategy == "adaptive":
            # Check if using GT variance debug mode
            if gt_variance_map is not None:
                # Debug mode: use ground truth variance for adaptive sampling
                loss_dict = psf_adaptive_sampling_step_debug(
                    model=model,
                    batch=batch,
                    num_mc_samples=args.num_mc_samples,
                    device=device,
                    beta=0.5,  # Default beta for debug
                )
            else:
                # Normal mode: use model-predicted variance
                loss_dict = psf_adaptive_sampling_step(
                    model=model,
                    batch=batch,
                    volume_shape=vol_norm.shape,
                    num_mc_samples=args.num_mc_samples,
                    device=device,
                )
        else:
            raise ValueError(f"Invalid sampling strategy: {args.sampling_strategy}")

        # Log the loss
        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item() if value is not None else 0, step)

        loss = loss_dict["total_loss"]
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)
        writer.add_scalar("train/GradClipped", float(grad_norm > 100.0), step)

        pbar.update(1)
        postfix_dict = {
            'recon_loss': f'{(loss_dict["reconstruction_loss"]).item():.6f}',
        }
        if current_alpha > 1e-5:
            postfix_dict['sp_alpha'] = f'{current_alpha:.4f}'
        if "clear_loss" in loss_dict:
            postfix_dict['clear_loss'] = f'{loss_dict["clear_loss"].item():.6f}'
        if "tv_loss" in loss_dict:
            postfix_dict['tv_loss'] = f'{loss_dict["tv_loss"].item():.6f}'
        if "cross_scale_loss" in loss_dict:
            postfix_dict['cross_loss'] = f'{loss_dict["cross_scale_loss"].item():.6f}'
        pbar.set_postfix(postfix_dict)
        
        # Early stopping check
        current_loss = loss_dict["reconstruction_loss"].item()
        if current_loss < best_loss - early_stop_min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at step {step} due to no improvement in reconstruction loss for {early_stop_patience} steps.")
            break
        
    
    pbar.close()

    
    # Save final model
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': model_config,
    }
    torch.save(save_obj, args.save_path)
    print(f"\nSaved final model to {args.save_path}")
    
    writer.close()


if __name__ == "__main__":
    main()
