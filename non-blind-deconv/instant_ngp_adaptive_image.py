import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
#import trimesh
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from models.instantngp import InstantNGPTorchModel
from sample_strategy.random_sample import sample_random_coords
#from sample_strategy.distribution_sample import sample_distribution_coords_blocked
from sample_strategy.distribution_sample import sample_importance_coords
from sample_strategy.sdf_sample import sample_sdf_coords
from regularization.TVs import compute_hyperlaplacian_tv

from visualize.utils import build_grid, gaussian_psf_from_cov_torch
from debug.adaptive import psf_adaptive_sampling_step_debug

class VolumeDataset(Dataset):
    def __init__(self, norm_volume_tensor, L, num_mc_samples, num_pixels_per_step, num_batches, clear_volume_tensor=None, gt_variance_tensor=None, psf_type='covariance', discrete_psf=None, mesh=None, cached_surface_points=None):
        self.norm_volume_tensor = norm_volume_tensor
        self.clear_volume_tensor = clear_volume_tensor
        self.gt_variance_tensor = gt_variance_tensor
        self.L = L.cpu() if L is not None else None  # Keep on CPU for parallel data loading
        self.num_mc_samples = num_mc_samples
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.volume_shape = norm_volume_tensor.shape
        self.psf_type = psf_type
        self.discrete_psf = discrete_psf.cpu() if discrete_psf is not None else None
        self.mesh = mesh
        self.cached_surface_points = cached_surface_points
        
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
        
        if self.psf_type == 'discrete':
            # Sample from discrete PSF
            # discrete_psf shape: (D, H, W) for 3D or (H, W) for 2D
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
        else:
            # Covariance-based sampling (original method)
            return torch.randn(sampling_budget, n_dims) @ self.L.T
    
    def _sample_sdf_coords(self, n):
        return sample_sdf_coords(
            mesh=self.mesh,
            n=n,
            volume_shape=self.volume_shape,
            device='cpu',
            cached_surface_points=self.cached_surface_points
        )
    
    def __getitem__(self, idx):
        if self.mesh is not None and len(self.volume_shape) == 3:
            target_coords = self._sample_sdf_coords(self.num_pixels_per_step)
        else:
            target_coords = sample_random_coords(self.num_pixels_per_step, self.volume_shape)
        
        # Get target values - handle both tensor and numpy array inputs
        # Convert indices to numpy for memory-mapped array access
        if len(self.volume_shape) == 3:
            z_indices = target_coords[:, 0].numpy()
            y_indices = target_coords[:, 1].numpy()
            x_indices = target_coords[:, 2].numpy()
            # Index into memory-mapped array, convert to native byte order, then to tensor
            values = self.norm_volume_tensor[z_indices, y_indices, x_indices]
            target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            if self.clear_volume_tensor is not None:
                clear_values = torch.from_numpy(np.ascontiguousarray(self.clear_volume_tensor[z_indices, y_indices, x_indices], dtype=np.float32))
            else:
                clear_values = None
            gt_variance_values = self.gt_variance_tensor[z_indices, y_indices, x_indices] if self.gt_variance_tensor is not None else None
        else:
            y_indices = target_coords[:, 0].numpy()
            x_indices = target_coords[:, 1].numpy()
            values = self.norm_volume_tensor[y_indices, x_indices]
            target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            if self.clear_volume_tensor is not None:
                clear_values = torch.from_numpy(np.ascontiguousarray(self.clear_volume_tensor[y_indices, x_indices], dtype=np.float32))
            else:
                clear_values = None
            gt_variance_values = self.gt_variance_tensor[y_indices, x_indices] if self.gt_variance_tensor is not None else None
        
        # Generate fresh offsets on every iteration (only if PSF is used)
        if self.L is not None or self.discrete_psf is not None:
            sampled_offsets = self._generate_offsets()
        
        target_coords_normalized = target_coords.float() * self.inv_shape
        
        # Normalize target_values from 0-65535 to 0-1
        target_values = target_values / 65535.0 * 100
        
        batch_dict = {
            'target_coords': target_coords_normalized,
            'target_values': target_values.float(),
        }
        
        # Only include sampled_offsets if PSF is used
        if self.L is not None or self.discrete_psf is not None:
            sampled_offsets_normalized = sampled_offsets.float() * self.inv_shape
            batch_dict['sampled_offsets'] = sampled_offsets_normalized
        
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
) -> torch.Tensor:
    # Get pre-computed data from DataLoader
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets = batch['sampled_offsets'].to(device, non_blocking=True)
    
    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    sampled_offsets = sampled_offsets.view(num_pixels, num_mc_samples, n_dims)
    
    # Anti-aliased integration: sample multiple points within each pixel
    if pixel_samples > 1 and pixel_scale is not None:
        # For each PSF sample, we now sample pixel_samples points within the pixel
        # Jitter within pixel: uniform in [-0.5, 0.5] * pixel_scale
        pixel_jitter = (torch.rand(num_pixels, num_mc_samples, pixel_samples, n_dims, device=device) - 0.5) * pixel_scale.to(device)
        
        # Expand source_coords to include pixel samples
        # source_coords: [num_pixels, num_mc_samples, n_dims]
        source_coords = target_coords.unsqueeze(1) + sampled_offsets
        source_coords = source_coords.unsqueeze(2) + pixel_jitter  # [num_pixels, num_mc_samples, pixel_samples, n_dims]
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
        # [num_pixels, num_mc_samples, pixel_samples] -> average over pixel_samples first, then mc_samples
        pred_samples = pred_flat.view(num_pixels, num_mc_samples, pixel_samples)
        pred_samples = pred_samples.mean(dim=2)  # Average over pixel samples (anti-aliasing)
    else:
        pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    
    # Simulate observed values by averaging over PSF samples
    simulated_values = pred_samples.mean(dim=1)
    total_loss = 0.0
    data_loss_mc = F.mse_loss(simulated_values.float(), target_values.float())

    #data_loss_mc = F.poisson_nll_loss(input=simulated_values.float(), target=target_values.float(), log_input=False, full=True, reduction="mean")

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
    pixel_samples: int = 1,  # Anti-aliasing: samples per pixel
    pixel_scale: torch.Tensor = None,  # For jittering within pixel
) -> torch.Tensor:
    """
    Simple training step with direct MSE loss (no PSF)
    Uses pre-sampled coordinates from DataLoader
    When pixel_samples > 1, integrates over pixel area for anti-aliased training
    """
    # Get pre-computed data from DataLoader
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    
    n_dims = target_coords.shape[1]
    num_pixels = target_coords.shape[0]
    
    # Anti-aliased integration: sample multiple points within each pixel
    if pixel_samples > 1 and pixel_scale is not None:
        # Jitter within pixel: uniform in [-0.5, 0.5] * pixel_scale
        pixel_jitter = (torch.rand(num_pixels, pixel_samples, n_dims, device=device) - 0.5) * pixel_scale.to(device)
        coords_expanded = target_coords.unsqueeze(1) + pixel_jitter  # [num_pixels, pixel_samples, n_dims]
        coords_expanded = torch.clamp(coords_expanded, 0.0, 1.0)
        coords_flat = coords_expanded.view(-1, n_dims)
    else:
        coords_flat = target_coords
    
    # Reorder coordinates for model input
    if n_dims == 3:
        # 3D: reorder to x, y, z (coords are in z, y, x order)
        coords_for_model = torch.stack([
            coords_flat[:, 2],  # x
            coords_flat[:, 1],  # y
            coords_flat[:, 0]   # z
        ], dim=-1)
    else:
        # 2D: reorder to x, y (coords are in y, x order)
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
    - E_model[x^2]: from MC samples of model prediction
    - E_input[x]:   from input blurry volume (target_values)
    """
    target_coords_normalized = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)  # blurry input at those coords
    sampled_offsets_normalized = batch['sampled_offsets'].to(device, non_blocking=True)

    n_dims = target_coords_normalized.shape[1]

    if n_dims == 3:
        # z,y,x -> x,y,z
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 2],
            target_coords_normalized[:, 1],
            target_coords_normalized[:, 0]
        ], dim=-1)
    else:
        # y,x -> x,y
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 1],
            target_coords_normalized[:, 0]
        ], dim=-1)

    _, log_variance = model(target_coords_for_model)  # shape [N, 1] or [N]
    log_variance = log_variance.view(-1)
    variance = torch.exp(log_variance)  # predicted variance per base point, > 0

    # compute adaptive MC budget per point
    sampling_budget = num_mc_samples * target_coords_normalized.shape[0]
    num_samples_per_data = sampling_budget * variance / (variance.sum() + eps)
    num_samples_per_data_floor = num_samples_per_data.floor()
    missing_samples = num_samples_per_data - num_samples_per_data_floor
    num_samples_per_data_floor = num_samples_per_data_floor.long()

    # at least 1 sample per data
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

    num_samples_per_data = num_samples_per_data_floor  # shape [N]

    # resample coords with offsets (MC)
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

    # model prediction on MC samples
    pred_flat, _ = model(x_resampled_for_model)  # shape [total_MCs]
    pred_flat = pred_flat.view(-1)

    # indices: MC sample -> base point index
    indices = torch.arange(
        target_coords_normalized.shape[0],
        device=target_coords_normalized.device
    )
    indices_resampled = torch.repeat_interleave(indices, num_samples_per_data, dim=0)

    # reconstruction loss: average MC samples per point
    pred_target_mc = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    pred_target_mc.scatter_add_(0, indices_resampled, pred_flat)
    pred_target_mc = pred_target_mc / num_samples_per_data.float()

    loss_reconstruction = F.mse_loss(pred_target_mc, target_values.float())

    # variance supervision:
    #     var_gt = E_model[x^2] - (E_input[x])^2
    # 7.1 E_model[x^2]: MC estimate per base point
    E_x2 = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    E_x2.scatter_add_(0, indices_resampled, pred_flat ** 2)
    E_x2 = E_x2 / num_samples_per_data.float()  # shape [N]

    # 7.2 E_input[x]: from blurry input volume at that point
    E_x = target_values.float()  # shape [N]

    var_gt = E_x2 - E_x ** 2
    #var_gt = var_gt.clamp_min(eps)  # 数值上防止负数 / log 问题

    pred_var = variance.view_as(var_gt)

    # var_gt_det = var_gt.detach()

    loss_variance = F.mse_loss(pred_var, var_gt)

    # total loss
    total_loss = loss_reconstruction * 100 + loss_variance

    return {
        "reconstruction_loss": loss_reconstruction,
        "variance_loss": loss_variance,
        "total_loss": total_loss,
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_npy", type=str, default="/workspace/combined_volumes.npy")
    parser.add_argument("--psf_covariance_path", type=str, default="/workspace/PSF.tif")
    parser.add_argument("--psf_type", type=str, default="discrete", choices=["covariance", "discrete"], help="Type of PSF: 'covariance' for covariance matrix (.pt) or 'discrete' for discrete PSF (.npy)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/largelsm.pth")
    parser.add_argument("--logdir", type=str, default="../runs/largelsm")
    parser.add_argument("--num_mc_samples", type=int, default=50, help="Number of PSF samples per pixel")
    parser.add_argument("--num_pixels_per_step", type=int, default=180000, help="Number of pixels per step")
    parser.add_argument("--sampling_strategy", type=str, default="uniform", choices=["uniform", "adaptive", "simple"])
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file (.obj, .ply, etc.) for SDF-based sampling (3D only)")
    parser.add_argument("--progressive_steps", type=int, default=3000, help="Number of steps to unlock progressively")
    
    parser.add_argument("--tv_loss_weight", type=float, default=None, help="Weight for total variation loss")
    
    # Stochastic training params
    parser.add_argument("--sp_alpha_init", type=float, default=0.03, 
                        help="Initial std dev for stochastic preconditioning")
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33, 
                        help="Fraction of training steps over which to decay alpha to 0 (Paper suggests ~1/3)")

    # adaptive sampling params
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for beta-nll loss")
    parser.add_argument("--beta-nll-multiplier-start", type=float, default=1e-8, help="Starting multiplier for beta-nll loss")
    parser.add_argument("--beta-nll-multiplier-end", type=float, default=1e-2, help="Ending multiplier for beta-nll loss")
    parser.add_argument("--warmup-steps", type=int, default=30, help="Number of warmup steps for beta-nll multiplier")
    parser.add_argument("--gt_variance_path", type=str, default=None, help="Path to ground truth variance .npy file for debug adaptive sampling")
    
    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Clear image regularization
    parser.add_argument("--clear_image_path", type=str, default=None, help="Path to clear/sharp image for L1 regularization")
    parser.add_argument("--clear_loss_weight", type=float, default=0.1, help="Weight for clear image L1 regularization loss")

    # Encoder config
    parser.add_argument("--num_levels", type=int, default=20, help="Number of hash encoding levels")
    parser.add_argument("--level_dim", type=int, default=2, help="Feature dimension per level")
    parser.add_argument("--base_resolution", type=int, default=16, help="Base grid resolution")
    parser.add_argument("--log2_hashmap_size", type=int, default=23, help="Log2 of hash table size")
    parser.add_argument("--desired_resolution", type=int, default=8192, help="Finest resolution")
    
    # Decoder config
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of MLP layers")
    
    # Anti-aliasing
    parser.add_argument("--pixel_samples", type=int, default=1, help="Number of samples per pixel for anti-aliased training (1=point sampling, 4=recommended for AA)")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load from npy using memory mapping for large volumes
    # IMPORTANT: Keep as memory-mapped NumPy array - do NOT convert to tensor!
    # Converting to tensor would load the entire 200GB into RAM
    vol_norm_np = np.load(args.volume_npy, mmap_mode='r')  # Read-only memory map
    print(f"Loaded volume with shape: {vol_norm_np.shape}, dtype: {vol_norm_np.dtype}")
    
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
        if clear_vol_norm.shape != vol_norm_np.shape:
            raise ValueError(f"Clear image shape {clear_vol_norm.shape} does not match blurred image shape {vol_norm_np.shape}")
        
        print(f"Clear image loaded with shape: {clear_vol_norm.shape}, normalized max: {clear_vol_norm.max():.4f}")
    
    # Load ground truth variance if provided (for debug adaptive sampling)
    gt_variance_map = None
    if args.gt_variance_path is not None:
        print(f"Loading ground truth variance from {args.gt_variance_path}")
        gt_variance_np = np.load(args.gt_variance_path).astype(np.float32)
        gt_variance_map = torch.from_numpy(gt_variance_np).float().to(device)
        
        # Verify dimensions match
        if gt_variance_map.shape != vol_norm_np.shape:
            raise ValueError(f"GT variance shape {gt_variance_map.shape} does not match blurred image shape {vol_norm_np.shape}")
        
        print(f"GT variance loaded with shape: {gt_variance_map.shape}")
        print(f"GT variance statistics: min={gt_variance_map.min():.6f}, max={gt_variance_map.max():.6f}, mean={gt_variance_map.mean():.6f}")
    
    # Detect 2D or 3D
    n_dims = len(vol_norm_np.shape)
    
    # Compute pixel scale for anti-aliased training (1 pixel in normalized coords)
    if n_dims == 3:
        pixel_scale = torch.tensor([1.0 / (vol_norm_np.shape[0] - 1), 
                                     1.0 / (vol_norm_np.shape[1] - 1), 
                                     1.0 / (vol_norm_np.shape[2] - 1)])
    else:
        pixel_scale = torch.tensor([1.0 / (vol_norm_np.shape[0] - 1), 
                                     1.0 / (vol_norm_np.shape[1] - 1)])
    print(f"Pixel scale for anti-aliasing: {pixel_scale}")
    if args.pixel_samples > 1:
        print(f"Anti-aliased training enabled: {args.pixel_samples} samples per pixel")
    
    # Load from checkpoint if specified
    if args.load_checkpoint is not None:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        
        # Load encoder and decoder configs from checkpoint (override command-line args)
        encoder_config = checkpoint['encoder_config']
        decoder_config = checkpoint['decoder_config']
        
        print("Loaded encoder config from checkpoint:")
        print(f"  {encoder_config}")
        print("Loaded decoder config from checkpoint:")
        print(f"  {decoder_config}")
    else:
        # Initialize model with tiny-cuda-nn configs from command-line args
        encoder_config = {
            "otype": "HashGrid",
            "n_levels": args.num_levels,
            "n_features_per_level": args.level_dim,
            "log2_hashmap_size": args.log2_hashmap_size,
            "base_resolution": args.base_resolution,
            "per_level_scale": np.exp((np.log(args.desired_resolution) - np.log(args.base_resolution)) / (args.num_levels - 1)),
        }
        
        decoder_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": args.hidden_dim,
            "n_hidden_layers": args.num_layers,
        }
    
    # Create model
    model = InstantNGPTorchModel(
        encoder_config=encoder_config, 
        decoder_config=decoder_config,
        n_input_dims=n_dims,
        learn_variance=False,
    ).to(device)
    
    # Load model state dict if checkpoint was provided
    if args.load_checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint")
    
    model.train()
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    n_levels = encoder_config["n_levels"]
    base = encoder_config["base_resolution"]
    per_level_scale = encoder_config["per_level_scale"]
    log2_hashmap_size = encoder_config["log2_hashmap_size"]
    hashmap_max = 2 ** log2_hashmap_size

    total_nodes = 0
    for i in range(n_levels):
        res = int(base * (per_level_scale ** i))
        nodes = min(hashmap_max, res ** 2 if n_dims == 2 else res ** 3)
        total_nodes += nodes

    print(f"  Total hashgrid nodes: {total_nodes:,}")
    
    
    # Optimizer and scheduler (use command-line lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    
    # Load optimizer state if checkpoint was provided
    if args.load_checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Starting with fresh optimizer state")
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.9)
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
    
    # Load PSF and precompute Cholesky decomposition for efficient sampling (only if not using simple strategy)
    L = None
    discrete_psf = None
    
    if args.sampling_strategy != "simple":
        if args.psf_type == "covariance":
            # Load covariance matrix
            sigma = torch.load(args.psf_covariance_path).to(device).float()
            print(f"PSF covariance matrix shape: {sigma.shape}")
            print(f"PSF covariance matrix:\n{sigma}")
            # Calculate and print PSF standard deviations (size)
            if sigma.dim() > 0:
                psf_std = torch.sqrt(torch.diag(sigma))
                print(f"PSF standard deviations (size) along each axis: {psf_std.cpu().numpy()}")
            # Verify PSF covariance dimensions match data dimensions
            if sigma.shape[0] != n_dims or sigma.shape[1] != n_dims:
                raise ValueError(f"PSF covariance shape {sigma.shape} does not match data dimensions {n_dims}D")
            L = torch.linalg.cholesky(sigma)  # Cholesky factor (will be on CPU in dataset)
        
        elif args.psf_type == "discrete":
            # Load discrete PSF from .npy or .tif file
            if args.psf_covariance_path.endswith(".tif") or args.psf_covariance_path.endswith(".tiff"):
                discrete_psf_np = tifffile.imread(args.psf_covariance_path).astype(np.float32)
            elif args.psf_covariance_path.endswith(".npy"):
                discrete_psf_np = np.load(args.psf_covariance_path).astype(np.float32)
            else:
                raise ValueError(f"Unsupported PSF file format: {args.psf_covariance_path}. Expected .tif, .tiff, or .npy")
            
            discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
            print(f"Discrete PSF shape: {discrete_psf.shape}")
            print(f"Discrete PSF min: {discrete_psf.min():.6f}, max: {discrete_psf.max():.6f}")
            
            # Verify PSF dimensions match data dimensions
            if len(discrete_psf.shape) != n_dims:
                raise ValueError(f"Discrete PSF dimensions {len(discrete_psf.shape)} does not match data dimensions {n_dims}D")
            
    else:
        # For simple strategy, PSF not needed
        print("Simple strategy: PSF not loaded")


    # Load mesh for SDF sampling if mesh_path is provided
    mesh = None
    cached_surface_points = None
    if args.mesh_path is not None:
        print(f"Loading mesh from {args.mesh_path} for SDF-based sampling...")
        print("Using SDF-based sampling strategy (bounds=1.2, uniform_ratio=0.25)")
    
    # Create dataset and dataloader for parallel data preparation
    dataset = VolumeDataset(
        norm_volume_tensor=vol_norm_np,  # Pass memory-mapped NumPy array directly
        L=L.cpu() if L is not None else None,
        num_mc_samples=args.num_mc_samples if args.sampling_strategy != "simple" else 1,
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
        clear_volume_tensor=clear_vol_norm,
        gt_variance_tensor=gt_variance_map.cpu() if gt_variance_map is not None else None,
        psf_type=args.psf_type,
        discrete_psf=discrete_psf.cpu() if discrete_psf is not None else None,
        mesh=mesh,
        cached_surface_points=cached_surface_points,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=12,  
        #pin_memory=True, 
        #persistent_workers=True, 
        #prefetch_factor=8, 
    )
    
    # Create iterator
    data_iter = iter(dataloader)
    
    # Progressive training configuration
    progressive_training = True  
    if progressive_training:
        n_levels = encoder_config["n_levels"]
        initial_levels = 4  
        # Linear schedule: unlock one level every (steps / (n_levels - initial_levels)) steps
        steps_per_level = args.progressive_steps // (n_levels - initial_levels)
        print(f"Progressive training enabled:")
        print(f"  Starting with {initial_levels} levels, unlocking 1 level every {steps_per_level} steps")
        print(f"  Total levels: {n_levels}")
        
    # Early stopping configuration
    early_stop_patience = 10000000  # Stop if no improvement for 50 steps
    early_stop_min_delta = 1e-6  # Minimum change to qualify as improvement
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


    
    # Track the last set level to avoid redundant set_max_level calls
    last_set_level = -1
    
    for step in range(args.steps):
        # Update progressive training level
        if progressive_training:
            current_level = min(initial_levels + (step // steps_per_level), n_levels)
            if current_level != last_set_level:
                model.set_max_level(current_level)
                last_set_level = current_level
                if step > 0: 
                    if current_level < n_levels:
                        print(f"\nStep {step}: Unlocked level {current_level}/{n_levels}")
                    else:
                        print(f"\nStep {step}: Unlocked level {n_levels}/{n_levels} (max)")
        
        batch = next(data_iter)
        batch = {k: v.squeeze(0) for k, v in batch.items()}
        
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
                    beta=args.beta,
                )
            else:
                # Normal mode: use model-predicted variance
                loss_dict = psf_adaptive_sampling_step(
                    model=model,
                    batch=batch,
                    volume_shape=vol_norm_np.shape,
                    num_mc_samples=args.num_mc_samples,
                    device=device,
                )
        else:
            raise ValueError(f"Invalid sampling strategy: {args.sampling_strategy}")

        # Log the loss
        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item() if value is not None else 0, step)

        # Calculate final loss
        if "variance_loss" in loss_dict:
            if step < args.warmup_steps:
                warmup_progress = step / args.warmup_steps
                current_multiplier = args.beta_nll_multiplier_start + \
                    (args.beta_nll_multiplier_end - args.beta_nll_multiplier_start) * warmup_progress
            else:
                current_multiplier = args.beta_nll_multiplier_end
            
            loss_dict["variance_loss"] = loss_dict["variance_loss"] * current_multiplier
        
        loss = loss_dict["total_loss"] # assume total_loss is always present
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)
        # Log if gradient was clipped (grad_norm > max_norm)
        writer.add_scalar("train/GradClipped", float(grad_norm > 10.0), step)

        pbar.update(1)
        postfix_dict = {
            'recon_loss': f'{(loss_dict["reconstruction_loss"]).item():.6f}',
            'lr': f'{current_lr:.2e}',
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
            print(f"\nearly stopping at step {step} due to no improvement in reconstruction loss for {early_stop_patience} steps.")
            break
        
    
    pbar.close()

    
    # Save final model
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }
    torch.save(save_obj, args.save_path)
    print(f"\nSaved final model to {args.save_path}")
    
    writer.close()


if __name__ == "__main__":
    main()

