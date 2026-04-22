"""
Total Variation (TV) utilities for computing and visualizing TV loss on neural fields.
"""

import torch
#import matplotlib.pyplot as plt
import cv2


def gaussian_psf_from_cov_torch(cov: torch.Tensor, trunc_sigma=4.0, device=None):
    """
    cov: [2,2] covariance on device
    return psf kernel: [1,1,Hk,Wk], sum=1
    """
    if device is None:
        device = cov.device
    cov = 0.5 * (cov + cov.T)
    inv_cov = torch.linalg.inv(cov + 1e-12 * torch.eye(2, device=device))

    stds = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
    radii = torch.clamp(torch.ceil(trunc_sigma * stds), min=1).long()
    ry, rx = radii[0].item(), radii[1].item()

    y = torch.arange(-ry, ry + 1, device=device)
    x = torch.arange(-rx, rx + 1, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([yy, xx], dim=-1).to(torch.float32)  # [Hk,Wk,2]

    # exp(-0.5 * g^T inv_cov g)
    expo = -0.5 * torch.einsum("...i,ij,...j->...", grid, inv_cov, grid)
    psf = torch.exp(expo)
    psf = psf / psf.sum()
    return psf[None, None, ...]  # [1,1,Hk,Wk]



def save_grad_magnitude(grad_mag, grid_res, filename):
    img = grad_mag.reshape(grid_res, grid_res).cpu().numpy()
    cv2.imwrite(filename + "original.png", img)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='turbo')
    plt.colorbar()
    plt.title("Gradient Magnitude")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def build_grid(n_dims, grid_res, device, data_shape=(512, 512)):
    """
    Build a regular grid of coordinates in [0, 1]^n_dims.
    
    Args:
        n_dims: Number of dimensions (2 or 3)
        grid_res: Resolution of the grid along each dimension
        device: PyTorch device
        data_shape: Optional tuple of data shape (y_max, x_max) for 2D or (z_max, y_max, x_max) for 3D.
                   If provided, limits grid to upper-left region (e.g., 4096x4096 for 2D).
        
    Returns:
        coords: [grid_res^n_dims, n_dims] tensor of coordinates in [0, 1] space
    """
    if n_dims == 2:
        # Determine the range for sampling - map to cropped region [13000:13000+4096, 32500:32500+4096]
        if data_shape is not None:
            y_max, x_max = data_shape
            # Crop region in absolute coordinates
            y_start, x_start = 0, 0
            y_limit = min(512, y_max)
            x_limit = min(512, x_max)
            
            # Normalize to [0, 1] based on full image dimensions
            y_start_norm = y_start / (y_max - 1) if y_max > 1 else 0.0
            x_start_norm = x_start / (x_max - 1) if x_max > 1 else 0.0
            y_end_norm = (y_start + y_limit) / (y_max - 1) if y_max > 1 else 1.0
            x_end_norm = (x_start + x_limit) / (x_max - 1) if x_max > 1 else 1.0
        else:
            y_start_norm = 0.0
            x_start_norm = 0.0
            y_end_norm = 1.0
            x_end_norm = 1.0
        
        y = torch.linspace(y_start_norm, y_end_norm, grid_res, device=device)
        x = torch.linspace(x_start_norm, x_end_norm, grid_res, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    
    elif n_dims == 3:
        
        # Full volume in normalized [0, 1] space
        z = torch.linspace(0.0, 1.0, grid_res, device=device)
        y = torch.linspace(0.0, 1.0, grid_res, device=device)
        x = torch.linspace(0.0, 1.0, grid_res, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    else:
        raise NotImplementedError(f"Only 2D and 3D are supported, got {n_dims}D")
    
    return coords


def visualize_autograd_tv(model, coords_grid):
    """
    Visualize TV using automatic differentiation (autograd).
    Computes the gradient magnitude at each point in coords_grid.
    
    Args:
        model: Neural network model with forward(coords, variance=None) method
        coords_grid: [N, D] tensor of coordinates in model's input space
        
    Returns:
        grad_mag: [N] tensor of gradient magnitudes
    """
    coords = coords_grid.clone().detach().requires_grad_(True)
    values, _ = model(coords, variance=None)

    grads = torch.autograd.grad(
        outputs=values,
        inputs=coords,
        grad_outputs=torch.ones_like(values),
        create_graph=False,
        retain_graph=False
    )[0]

    # 梯度模长
    grad_mag = grads.norm(dim=-1)        

    return grad_mag.detach()


def visualize_fd_tv(model, coords_grid, eps=1e-3):
    """
    Visualize TV using finite differences.
    Computes approximate gradient magnitude at each point in coords_grid.
    
    Args:
        model: Neural network model with forward(coords, variance=None) method
        coords_grid: [N, D] tensor of coordinates in model's input space
        eps: Finite difference step size
        
    Returns:
        grad_mag: [N] tensor of gradient magnitudes
    """
    N, D = coords_grid.shape
    device = coords_grid.device
    
    # 原位置 f(x)
    values, _ = model(coords_grid, variance=None)
    values = values.squeeze(-1)

    shift_vecs = torch.eye(D, device=device) * eps   # [D, D]

    grad_list = []
    for d in range(D):
        coords_shifted = coords_grid + shift_vecs[d]   # x+eps_d
        values_shift, _ = model(coords_shifted, variance=None)
        grad_d = (values_shift.squeeze(-1) - values) / eps
        grad_list.append(grad_d)

    grads = torch.stack(grad_list, dim=-1)    # [N, D]
    grad_mag = grads.norm(dim=-1)             # [N]

    return grad_mag.detach()
