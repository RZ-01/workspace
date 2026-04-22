import torch
import torch.nn.functional as F
import torch.nn as nn

def psf_adaptive_sampling_step(
    model: nn.Module,
    batch: dict,
    volume_shape: tuple,
    num_mc_samples: int,
    device: torch.device,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Training step with PSF-based Monte Carlo sampling and FP16 mixed precision using adaptive sampling.
    Data is pre-computed in DataLoader for parallel processing.
    Supports both 2D and 3D data.
    """
    # Get pre-computed data from DataLoader (already normalized to [0, 1])
    target_coords_normalized = batch['target_coords'].to(device, non_blocking=True)  # Already normalized
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets_normalized = batch['sampled_offsets'].to(device, non_blocking=True)  # Already normalized
    
    n_dims = target_coords_normalized.shape[1]
    
    # Reorder coordinates for model input
    if n_dims == 3:
        # 3D: reorder to x, y, z (coords are in z, y, x order)
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 2],  # x
            target_coords_normalized[:, 1],  # y
            target_coords_normalized[:, 0]   # z
        ], dim=-1)
    else:
        # 2D: reorder to x, y (coords are in y, x order)
        target_coords_for_model = torch.stack([
            target_coords_normalized[:, 1],  # x
            target_coords_normalized[:, 0]   # y
        ], dim=-1)
    
    # Forward pass
    _, log_variance = model(target_coords_for_model)

    # Compute the variance
    variance = torch.exp(log_variance)
    
    sampling_budget = num_mc_samples * target_coords_normalized.shape[0]
    num_samples_per_data = sampling_budget * \
        variance / variance.sum(dim=-1, keepdim=True)
    num_samples_per_data_floor = num_samples_per_data.floor()
    missing_samples = num_samples_per_data - num_samples_per_data_floor
    num_samples_per_data_floor = num_samples_per_data_floor.long()

    # Ensure at least 1 sample per data point
    num_samples_per_data_floor = torch.clamp(num_samples_per_data_floor, min=1)

    # Calculate budget difference (positive if under, negative if over)
    total_samples = num_samples_per_data_floor.sum().item()
    budget_diff = sampling_budget - total_samples

    # Adjust based on whether we're over or under budget
    # we are always under budget, is this necessary?
    abs_budget_diff = abs(budget_diff)
    adjustment_sign = 1 if budget_diff > 0 else -1

    # Select indices: fractional parts if under budget, sample counts if over
    selection_values = missing_samples if budget_diff > 0 else num_samples_per_data_floor.float()
    # Clamp to valid range to avoid topk index out of range error
    abs_budget_diff = int(min(abs_budget_diff, selection_values.shape[0]))
    
    if abs_budget_diff > 0:
        _, topk_indices = selection_values.topk(abs_budget_diff, dim=-1, sorted=False)
        # Apply adjustment
        num_samples_per_data_floor[topk_indices] += adjustment_sign

    num_samples_per_data = num_samples_per_data_floor

    # Calculate sampling locations (both coords and offsets are already normalized)
    x_resampled_normalized = torch.repeat_interleave(target_coords_normalized, num_samples_per_data, dim=0) + sampled_offsets_normalized
    
    # Clip to [0, 1] after adding MC perturbations
    x_resampled_normalized = torch.clamp(x_resampled_normalized, 0.0, 1.0)
    
    # Reorder coordinates for model input
    if n_dims == 3:
        # 3D: reorder to x, y, z
        x_resampled_for_model = torch.stack([
            x_resampled_normalized[:, 2],  # x
            x_resampled_normalized[:, 1],  # y
            x_resampled_normalized[:, 0]   # z
        ], dim=-1)
    else:
        # 2D: reorder to x, y
        x_resampled_for_model = torch.stack([
            x_resampled_normalized[:, 1],  # x
            x_resampled_normalized[:, 0]   # y
        ], dim=-1)

    # Call the neural field
    pred_flat, _ = model(x_resampled_for_model)

    # Repeat indices
    indices = torch.arange(target_coords_normalized.shape[0], device=target_coords_normalized.device)
    indices_resampled = torch.repeat_interleave(indices, num_samples_per_data, dim=0)

    # Repeat log_variance
    log_variance_resampled = torch.repeat_interleave(log_variance, num_samples_per_data, dim=0)
    variance_resampled = torch.repeat_interleave(variance + eps, num_samples_per_data, dim=0)

    # Repeat target
    target_values_resampled = torch.repeat_interleave(target_values, num_samples_per_data, dim=0)

    # Calculate the beta-nll loss
    multiplicative_factor = variance_resampled.detach() ** beta
    # Variance regularization
    loss_variance_reg = (0.5 * log_variance_resampled * multiplicative_factor).mean()
    # Compute the per-sample loss:
    # loss_variance_target= (0.5 * ((target_values_resampled.detach() - pred_flat) ** 2) * (multiplicative_factor / variance_resampled)).mean()
    loss_variance_target= (0.5 * ((target_values_resampled.detach() - pred_flat.detach()) ** 2) * (multiplicative_factor / variance_resampled)).mean()
    # Total beta-nll loss
    loss_beta_nll = loss_variance_reg + loss_variance_target

    # Reconstruction loss
    pred_target_mc = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    pred_target_mc.scatter_add_(0, indices_resampled, pred_flat)
    pred_target_mc = pred_target_mc / num_samples_per_data.float()
    loss_reconstruction = F.mse_loss(pred_target_mc, target_values.float())

    return {"reconstruction_loss": loss_reconstruction, "variance_loss": loss_beta_nll, "total_loss": loss_reconstruction + loss_beta_nll}

def psf_adaptive_sampling_step_debug(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
    beta: float = 1.0,
) -> torch.Tensor:
    target_coords_normalized = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets_normalized = batch['sampled_offsets'].to(device, non_blocking=True)
    gt_variance_values = batch['gt_variance_values'].to(device, non_blocking=True)
    
    n_dims = target_coords_normalized.shape[1]
    
    # Use pre-sampled GT variance (already in correct format)
    variance = gt_variance_values  # Already 1D tensor from DataLoader
    
    sampling_budget = num_mc_samples * target_coords_normalized.shape[0]
    num_samples_per_data = sampling_budget * \
        variance / variance.sum()
    num_samples_per_data_floor = num_samples_per_data.floor()
    missing_samples = num_samples_per_data - num_samples_per_data_floor
    num_samples_per_data_floor = num_samples_per_data_floor.long()

    # Ensure at least 1 sample per data point
    num_samples_per_data_floor = torch.clamp(num_samples_per_data_floor, min=1)

    # Calculate budget difference (positive if under, negative if over)
    total_samples = num_samples_per_data_floor.sum().item()
    budget_diff = sampling_budget - total_samples

    # Adjust based on whether we're over or under budget
    abs_budget_diff = abs(budget_diff)
    adjustment_sign = 1 if budget_diff > 0 else -1

    # Select indices: fractional parts if under budget, sample counts if over
    selection_values = missing_samples if budget_diff > 0 else num_samples_per_data_floor.float()
    
    if abs_budget_diff > 0 and selection_values.numel() > 0:
        selection_values_1d = selection_values.squeeze()
        _, topk_indices = selection_values_1d.topk(abs_budget_diff, sorted=False)
        # Apply adjustment
        num_samples_per_data_floor[topk_indices] += adjustment_sign

    num_samples_per_data = num_samples_per_data_floor

    # Calculate sampling locations (both coords and offsets are already normalized)
    x_resampled_normalized = torch.repeat_interleave(target_coords_normalized, num_samples_per_data, dim=0) + sampled_offsets_normalized
    
    # Clip to [0, 1] after adding MC perturbations
    x_resampled_normalized = torch.clamp(x_resampled_normalized, 0.0, 1.0)
    
    # Reorder coordinates for model input
    x_resampled_for_model = torch.stack([
        x_resampled_normalized[:, 1],  # x
        x_resampled_normalized[:, 0]   # y
    ], dim=-1)

    # Call the neural field (only get prediction, ignore model's variance output)
    pred_flat, _ = model(x_resampled_for_model)

    # Repeat indices for averaging predictions back to target pixels
    indices = torch.arange(target_coords_normalized.shape[0], device=target_coords_normalized.device)
    indices_resampled = torch.repeat_interleave(indices, num_samples_per_data, dim=0)

    # Reconstruction loss: average predictions and compare with target values
    pred_target_mc = torch.zeros_like(target_values, dtype=pred_flat.dtype)
    pred_target_mc.scatter_add_(0, indices_resampled, pred_flat)
    pred_target_mc = pred_target_mc / num_samples_per_data.float()
    loss_reconstruction = F.mse_loss(pred_target_mc, target_values.float()) * 100

    return {"reconstruction_loss": loss_reconstruction, "total_loss": loss_reconstruction}