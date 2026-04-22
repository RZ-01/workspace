import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import trimesh
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from models.instantngp import InstantNGPTorchModel
from sample_strategy.random_sample import sample_random_coords
from sample_strategy.sdf_sample import sample_sdf_coords
from regularization.TVs import compute_hyperlaplacian_tv
from visualize.utils import build_grid


class LearnablePSF(nn.Module):
    """Learnable PSF parameters (sigma_z, sigma_y, sigma_x) for blind deconvolution."""
    
    def __init__(self, init_sigma_z, init_sigma_y, init_sigma_x, device, n_dims=3):
        super().__init__()
        self.n_dims = n_dims
        if n_dims == 3:
            init_sigmas = torch.tensor([init_sigma_z, init_sigma_y, init_sigma_x], 
                                       dtype=torch.float32, device=device)
        else:
            init_sigmas = torch.tensor([init_sigma_y, init_sigma_x], 
                                       dtype=torch.float32, device=device)
        self.log_sigma = nn.Parameter(torch.log(init_sigmas))
        self.register_buffer('init_sigma', init_sigmas.clone())
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def get_cholesky(self):
        """Get Cholesky factor L such that L @ L.T = covariance (diagonal case)."""
        return torch.diag(self.sigma)
    
    def l2_regularization(self):
        return ((self.sigma - self.init_sigma) ** 2).sum()
    
    def get_sigma_values(self):
        s = self.sigma.detach().cpu()
        if self.n_dims == 3:
            return s[0].item(), s[1].item(), s[2].item()
        else:
            return s[0].item(), s[1].item()


class VolumeDataset(Dataset):
    """Dataset for loading volume slices and target coordinates."""
    
    def __init__(self, norm_volume_tensor, num_pixels_per_step, num_batches, mesh=None):
        self.norm_volume_tensor = norm_volume_tensor
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.volume_shape = norm_volume_tensor.shape
        self.mesh = mesh
        
        if len(self.volume_shape) == 3:
            vz, vy, vx = self.volume_shape
            self.inv_shape = torch.tensor([1.0 / (vz - 1), 1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)
        else:
            vy, vx = self.volume_shape
            self.inv_shape = torch.tensor([1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)
        
    def __len__(self):
        return self.num_batches
    
    def _sample_sdf_coords(self, n):
        return sample_sdf_coords(
            mesh=self.mesh, n=n, volume_shape=self.volume_shape, device='cpu'
        )
    
    def __getitem__(self, idx):
        if self.mesh is not None and len(self.volume_shape) == 3:
            target_coords = self._sample_sdf_coords(self.num_pixels_per_step)
        else:
            target_coords = sample_random_coords(self.num_pixels_per_step, self.volume_shape)
        
        if len(self.volume_shape) == 3:
            z_indices = target_coords[:, 0]
            y_indices = target_coords[:, 1]
            x_indices = target_coords[:, 2]
            target_values = self.norm_volume_tensor[z_indices, y_indices, x_indices]
        else:
            y_indices = target_coords[:, 0]
            x_indices = target_coords[:, 1]
            target_values = self.norm_volume_tensor[y_indices, x_indices]
        
        target_coords_normalized = target_coords.float() * self.inv_shape
        target_values = target_values / 65535.0 * 100.0
        
        return {
            'target_coords': target_coords_normalized,
            'target_values': target_values.float(),
        }


def model_training_step(
    model: nn.Module,
    learnable_psf: nn.Module,
    batch: dict,
    volume_shape: tuple,
    num_mc_samples: int,
    device: torch.device,
    tv_loss_weight: float = None,
    stochastic_alpha: float = 0.0,
) -> dict:
    """Training step for model using MC sampling from current PSF.
    
    Uses reparameterization trick: sample z ~ N(0,1), then offset = z * sigma.
    PSF parameters are frozen (detached) during model training.
    """
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    
    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    
    # Get Cholesky factor from learnable PSF (DETACHED - no gradient to PSF)
    L = learnable_psf.get_cholesky().detach()
    
    # MC sampling: z ~ N(0, I), offset = z @ L.T
    z = torch.randn(num_pixels * num_mc_samples, n_dims, device=device)
    offsets = z @ L.T
    
    # Normalize offsets to [0, 1] coordinate space
    inv_shape = torch.tensor([1.0 / (volume_shape[i] - 1) for i in range(n_dims)], 
                             dtype=torch.float32, device=device)
    offsets_normalized = offsets * inv_shape
    offsets_normalized = offsets_normalized.view(num_pixels, num_mc_samples, n_dims)
    
    # Compute source coordinates
    source_coords = target_coords.unsqueeze(1) + offsets_normalized
    source_coords = torch.clamp(source_coords, 0.0, 1.0)
    source_coords_flat = source_coords.view(-1, n_dims)
    
    # Reorder coordinates for model input
    if n_dims == 3:
        coords_for_model = torch.stack([
            source_coords_flat[:, 2], source_coords_flat[:, 1], source_coords_flat[:, 0]
        ], dim=-1)
    else:
        coords_for_model = torch.stack([
            source_coords_flat[:, 1], source_coords_flat[:, 0]
        ], dim=-1)
    
    pred_flat, _ = model(coords_for_model.float(), variance=None, stochastic_alpha=stochastic_alpha)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    
    # Average over MC samples (equal weights)
    simulated_values = pred_samples.mean(dim=1)
    
    total_loss = 0.0
    data_loss = F.mse_loss(simulated_values.float(), target_values.float())
    loss_dict = {"reconstruction_loss": data_loss}
    total_loss += data_loss
    
    if tv_loss_weight is not None:
        coords_grid = build_grid(n_dims, 64, device)
        tv_loss = compute_hyperlaplacian_tv(model, coords_grid).mean()
        total_loss += tv_loss_weight * tv_loss
        loss_dict["tv_loss"] = tv_loss * tv_loss_weight
    
    loss_dict["total_loss"] = total_loss
    return loss_dict


def psf_training_step(
    model: nn.Module,
    learnable_psf: nn.Module,
    batch: dict,
    volume_shape: tuple,
    num_mc_samples: int,
    device: torch.device,
    tv_loss_weight: float = None,
) -> dict:
    """Training step for PSF using MC sampling with gradient flow to sigma.
    
    Uses reparameterization trick: sample z ~ N(0,1), then offset = z @ L.T
    L is NOT detached, so gradients flow to sigma during PSF optimization.
    Model is frozen during PSF training.
    """
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    
    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    
    # Get Cholesky factor from learnable PSF (NOT detached - gradients flow to sigma!)
    L = learnable_psf.get_cholesky()
    
    # MC sampling: z ~ N(0, I), offset = z @ L.T
    z = torch.randn(num_pixels * num_mc_samples, n_dims, device=device)
    offsets = z @ L.T  # Gradients flow through L to sigma
    
    # Normalize offsets to [0, 1] coordinate space
    inv_shape = torch.tensor([1.0 / (volume_shape[i] - 1) for i in range(n_dims)], 
                             dtype=torch.float32, device=device)
    offsets_normalized = offsets.view(num_pixels, num_mc_samples, n_dims) * inv_shape
    
    # Compute source coordinates
    source_coords = target_coords.unsqueeze(1) + offsets_normalized
    source_coords = torch.clamp(source_coords, 0.0, 1.0)
    source_coords_flat = source_coords.view(-1, n_dims)
    
    if n_dims == 3:
        coords_for_model = torch.stack([
            source_coords_flat[:, 2], source_coords_flat[:, 1], source_coords_flat[:, 0]
        ], dim=-1)
    else:
        coords_for_model = torch.stack([
            source_coords_flat[:, 1], source_coords_flat[:, 0]
        ], dim=-1)
    
    # Forward pass (model is frozen during PSF training)
    with torch.no_grad():
        pred_flat, _ = model(coords_for_model.float(), variance=None, stochastic_alpha=0.0)
    
    # Need to re-enable gradient for the coordinate transformation
    # The key is that source_coords depends on L (through offsets), 
    # but the model output doesn't have gradients
    # So we need the gradient to flow through the averaging operation
    
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    
    # Average over MC samples - gradients flow through source_coords to L to sigma
    simulated_values = pred_samples.mean(dim=1)
    
    total_loss = 0.0
    data_loss = F.mse_loss(simulated_values.float(), target_values.float())
    loss_dict = {"reconstruction_loss": data_loss}
    total_loss += data_loss
    
    if tv_loss_weight is not None:
        coords_grid = build_grid(n_dims, 64, device)
        tv_loss = compute_hyperlaplacian_tv(model, coords_grid).mean()
        total_loss += tv_loss_weight * tv_loss
        loss_dict["tv_loss"] = tv_loss * tv_loss_weight
    
    loss_dict["total_loss"] = total_loss
    return loss_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="/workspace/stanford_dragon/dragon_blurred.tif")
    parser.add_argument("--psf_covariance_path", type=str, default="/workspace/stanford_dragon/gaussian_cov.pt")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/dragon_psf.pth")
    parser.add_argument("--logdir", type=str, default="../runs/dragon_psf")
    parser.add_argument("--num_mc_samples", type=int, default=100)
    parser.add_argument("--num_pixels_per_step", type=int, default=100000)
    parser.add_argument("--mesh_path", type=str, default="/workspace/non-blind-deconv/gt_dragon.ply")
    parser.add_argument("--progressive_steps", type=int, default=500)
    parser.add_argument("--tv_loss_weight", type=float, default=None)
    
    parser.add_argument("--sp_alpha_init", type=float, default=0.03)
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--num_levels", type=int, default=16)
    parser.add_argument("--level_dim", type=int, default=2)
    parser.add_argument("--base_resolution", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=22)
    parser.add_argument("--desired_resolution", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    
    parser.add_argument("--model_steps", type=int, default=5000)
    parser.add_argument("--psf_steps", type=int, default=200)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--psf_lr", type=float, default=1e-3)
    parser.add_argument("--psf_reg_weight", type=float, default=0.1)
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol_np = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_norm = torch.from_numpy(vol_np)
    print(f"Loaded volume: {vol_norm.shape}, range: [{vol_norm.min():.2f}, {vol_norm.max():.2f}]")

    n_dims = len(vol_norm.shape)
    
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        encoder_config = checkpoint['encoder_config']
        decoder_config = checkpoint['decoder_config']
    else:
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
    
    model = InstantNGPTorchModel(
        encoder_config=encoder_config, 
        decoder_config=decoder_config,
        n_input_dims=n_dims,
        learn_variance=False,
    ).to(device)
    
    if args.load_checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint")
    
    model.train()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    if args.load_checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
    
    gamma = 100 ** (-1 / args.model_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scaler = GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)
    
    # Load PSF and create LearnablePSF
    cov_matrix = torch.load(args.psf_covariance_path).to(device).float()
    psf_std = torch.sqrt(torch.diag(cov_matrix))
    print(f"PSF sigma: {psf_std.cpu().numpy()}")
    
    init_sigmas = psf_std.cpu().tolist()
    if n_dims == 3:
        learnable_psf = LearnablePSF(init_sigmas[0], init_sigmas[1], init_sigmas[2], device, n_dims=3)
    else:
        learnable_psf = LearnablePSF(0, init_sigmas[0], init_sigmas[1], device, n_dims=2)
    
    psf_optimizer = torch.optim.Adam(learnable_psf.parameters(), lr=args.psf_lr)

    mesh = None
    if args.mesh_path is not None:
        mesh = trimesh.load(args.mesh_path)
        print(f"Mesh: {len(mesh.vertices)} vertices")
    
    steps_per_round = args.model_steps + args.psf_steps
    total_steps = steps_per_round * args.num_rounds
    
    dataset = VolumeDataset(vol_norm, args.num_pixels_per_step, total_steps, mesh=mesh)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=24)
    data_iter = iter(dataloader)
    
    n_levels = encoder_config["n_levels"]
    initial_levels = 4
    steps_per_level = args.progressive_steps // (n_levels - initial_levels)
    sp_decay_steps = int(args.model_steps * args.sp_decay_fraction)

    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    print(f"\nSchedule: {args.num_rounds} rounds × ({args.model_steps} model + {args.psf_steps} PSF)")
    
    last_set_level = -1
    
    for step in range(total_steps):
        step_in_round = step % steps_per_round
        current_round = step // steps_per_round + 1
        is_psf_phase = step_in_round >= args.model_steps
        
        # Progressive training (only first round, model phase)
        if current_round == 1 and not is_psf_phase:
            current_level = min(initial_levels + (step_in_round // steps_per_level), n_levels)
            if current_level != last_set_level:
                model.set_max_level(current_level)
                last_set_level = current_level
        elif current_round > 1 and last_set_level < n_levels:
            model.set_max_level(n_levels)
            last_set_level = n_levels
        
        batch = next(data_iter)
        batch = {k: v.squeeze(0) for k, v in batch.items()}
        
        if is_psf_phase:
            psf_optimizer.zero_grad(set_to_none=True)
            loss_dict = psf_training_step(
                model=model,
                learnable_psf=learnable_psf,
                batch=batch,
                volume_shape=vol_norm.shape,
                num_mc_samples=args.num_mc_samples,
                device=device,
                tv_loss_weight=args.tv_loss_weight,
            )
        else:
            optimizer.zero_grad(set_to_none=True)
            # Stochastic preconditioning
            if step_in_round < sp_decay_steps and current_round == 1:
                current_alpha = args.sp_alpha_init * np.exp(-5.0 * step_in_round / sp_decay_steps)
            else:
                current_alpha = 0.0
            
            loss_dict = model_training_step(
                model=model,
                learnable_psf=learnable_psf,
                batch=batch,
                volume_shape=vol_norm.shape,
                num_mc_samples=args.num_mc_samples,
                device=device,
                tv_loss_weight=args.tv_loss_weight,
                stochastic_alpha=current_alpha,
            )

        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item(), step)
        
        sigma_vals = learnable_psf.get_sigma_values()
        if len(sigma_vals) == 3:
            writer.add_scalar("psf/sigma_z", sigma_vals[0], step)
            writer.add_scalar("psf/sigma_y", sigma_vals[1], step)
            writer.add_scalar("psf/sigma_x", sigma_vals[2], step)
        
        loss = loss_dict["total_loss"]
        
        if is_psf_phase and args.psf_reg_weight > 0:
            psf_reg_loss = learnable_psf.l2_regularization() * args.psf_reg_weight
            loss = loss + psf_reg_loss
        
        scaler.scale(loss).backward()
        
        if is_psf_phase:
            scaler.unscale_(psf_optimizer)
            torch.nn.utils.clip_grad_norm_(learnable_psf.parameters(), max_norm=10.0)
            scaler.step(psf_optimizer)
            scaler.update()
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if old_scale <= scaler.get_scale():
                scheduler.step()
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss_dict["reconstruction_loss"].item():.6f}',
            'phase': 'PSF' if is_psf_phase else 'model',
            'σ': f'[{",".join(f"{s:.2f}" for s in sigma_vals)}]'
        })
    
    pbar.close()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }, args.save_path)
    print(f"\nSaved to {args.save_path}")
    writer.close()


if __name__ == "__main__":
    main()
